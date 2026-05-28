import cv2
import numpy as np
import os
import threading
import urllib.request
import mediapipe as mp

# ── Model ──────────────────────────────────────────────────────────────────────
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hair_segmenter.tflite")
_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite"

HAIR_CATEGORY: int = 1

def _clip_beard(mask: np.ndarray, landmarks, h: int) -> np.ndarray:
    """
    Zero out hair mask pixels below the nose-mouth midpoint.
    Uses landmark 164 (under-nose/philtrum base) and 17 (chin top) to find
    the beard boundary — robust to head tilt and face scale.
    """
    # Landmark 164 = base of philtrum (just above lip), 13 = upper lip centre
    # Using the average of both gives a stable "top of beard" line
    philtrum = landmarks[164]
    upper_lip = landmarks[13]
    beard_top_y = int((philtrum[1] + upper_lip[1]) / 2)

    # Feather the cutoff so it's not a hard edge
    feather = max(5, int((landmarks[152][1] - landmarks[10][1]) * 0.04))
    clipped = mask.copy()

    # Hard zero below the cutoff
    clipped[beard_top_y + feather:, :] = 0.0

    # Smooth transition in the feather zone
    if feather > 0:
        for row in range(beard_top_y, min(beard_top_y + feather, h)):
            t = (row - beard_top_y) / feather          # 0 → 1
            clipped[row, :] *= (1.0 - t * t)           # quadratic fade

    return clipped


def _ensure_model() -> str:
    if not os.path.exists(_MODEL_PATH) or os.path.getsize(_MODEL_PATH) == 0:
        if os.path.exists(_MODEL_PATH):
            os.remove(_MODEL_PATH)  # remove empty file
        print(f"[hair_detection] Downloading HairSegmenter model...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[hair_detection] Download complete.")
    return _MODEL_PATH


# ── Legacy compatibility ───────────────────────────────────────────────────────
def eps_criterion(pixel, sensitivity: int = 240) -> bool:
    b, g, r = int(pixel[0]), int(pixel[1]), int(pixel[2])
    return (b + g + r) > sensitivity


# ── Main class ─────────────────────────────────────────────────────────────────
class Hair:
    """
    Non-blocking hair detector.

    The MediaPipe segmenter runs in a dedicated background thread so it never
    stalls the camera loop. The main thread always gets the *last completed*
    result instantly — stale by at most one inference cycle (~57 ms), which is
    completely imperceptible for a hair-coverage check.

    """

    def __init__(self):

        self._last_mask: np.ndarray | None = None
        self._last_ratio: float = 0.0
        self._lock = threading.Lock()

        self._pending_frame: np.ndarray | None = None
        self._pending_mask: np.ndarray | None = None
        self._frame_event = threading.Event()
        self._stop_event = threading.Event()

        # ── Build segmenter ───────────────────────────────────────────────────
        model_path = _ensure_model()

        BaseOptions = mp.tasks.BaseOptions
        ImageSegmenter = mp.tasks.vision.ImageSegmenter
        ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=model_path), # where is .tflite file
            running_mode=VisionRunningMode.IMAGE,
            output_category_mask=True, # true = uint8
            output_confidence_masks=False, # true = float32
        )
        self._segmenter = ImageSegmenter.create_from_options(options)

        # ── Start background worker ───────────────────────────────────────────
        self._thread = threading.Thread(target=self._worker, daemon=True, name="HairSegWorker")
        self._thread.start()

    # ── Background worker ─────────────────────────────────────────────────────

    def _worker(self):
        """Runs in a daemon thread. Processes frames as fast as the model allows."""
        while not self._stop_event.is_set():
            # Block until a frame is queued (or we're asked to stop)
            self._frame_event.wait()
            self._frame_event.clear()

            if self._stop_event.is_set():
                break

            with self._lock:
                frame = self._pending_frame
                face_mask = self._pending_mask
                self._pending_frame = None
                self._pending_mask = None

            if frame is None:
                continue

            try:
                hair_mask = self._run_segmenter(frame)
                ratio = self._compute_ratio(hair_mask, face_mask)
            except Exception as e:
                print(f"[HairSegWorker] inference error: {e}")
                continue

            # Publish results
            with self._lock:
                self._last_mask = hair_mask
                self._last_ratio = ratio

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _run_segmenter(self, image: np.ndarray) -> np.ndarray:
        """Run MediaPipe and return a binary float32 hair mask (H×W)."""
        h, w = image.shape[:2]
        rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=rgba)
        result = self._segmenter.segment(mp_image)

        if result.category_mask is None:
            return np.zeros((h, w), dtype=np.float32)

        cat = result.category_mask.numpy_view()   # uint8, shape (H, W)
        if cat.shape[:2] != (h, w):
            cat = cv2.resize(cat, (w, h), interpolation=cv2.INTER_NEAREST)

        return (cat == HAIR_CATEGORY).astype(np.float32)

    @staticmethod
    def _compute_ratio(hair_mask: np.ndarray, face_roi_mask: np.ndarray | None) -> float:
        if face_roi_mask is None:
            return 0.0
        face_region = face_roi_mask > 0
        if not np.any(face_region):
            return 0.0
        hair_in_face = hair_mask[face_region]
        return float(np.sum(hair_in_face)) / hair_in_face.size

    # ── Public API (all return instantly — zero blocking) ─────────────────────

    def get_hair_ratio(self, image: np.ndarray, face_roi_mask: np.ndarray) -> float:
        """
        Queue a frame for background inference and immediately return the last
        known hair ratio.  Never blocks the caller.
        """
        # Queue the new frame (overwrite any unprocessed one — only newest matters)
        with self._lock:
            self._pending_frame = image.copy()
            self._pending_mask = face_roi_mask.copy()

        self._frame_event.set()   # wake the worker

        # Return the last completed result right now
        with self._lock:
            return self._last_ratio

    def get_boolean_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Queue a frame for inference and return the last known binary mask.
        Falls back to a zero mask on the very first call before any result is ready.
        """
        with self._lock:
            self._pending_frame = image.copy()
            self._pending_mask = None

        self._frame_event.set()

        with self._lock:
            if self._last_mask is not None:
                return self._last_mask.copy()
        h, w = image.shape[:2]
        return np.zeros((h, w), dtype=np.float32)

    def apply_mask_on_image(self, image: np.ndarray) -> np.ndarray:
        """Debug overlay — paints hair pixels red. Uses the last completed mask."""
        with self._lock:
            mask = self._last_mask

        annotated = image.copy()
        if mask is not None and mask.shape[:2] == image.shape[:2]:
            annotated[mask.astype(bool)] = (0, 0, 255)
        return annotated

    def process(self, image: np.ndarray) -> tuple[int, int, int]:
        """Return median (R, G, B) of hair pixels using the last completed mask."""
        with self._lock:
            mask = self._last_mask

        if mask is None:
            return 0, 0, 0
        bool_mask = mask.astype(bool)
        if not np.any(bool_mask):
            return 0, 0, 0
        b, g, r = cv2.split(image)
        return int(np.median(r[bool_mask])), int(np.median(g[bool_mask])), int(np.median(b[bool_mask]))

    def __del__(self):
        try:
            self._stop_event.set()
            self._frame_event.set()   # unblock the worker so it can exit
            self._thread.join(timeout=2.0)
            self._segmenter.close()
        except Exception:
            pass

    def get_hair_mask_sync(self, image: np.ndarray, landmarks=None) -> np.ndarray:
        h, w = image.shape[:2]
        rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=rgba)
        result = self._segmenter.segment(mp_image)
        if result.category_mask is None:
            return np.zeros((h, w), dtype=np.float32)
        cat = result.category_mask.numpy_view()
        if cat.shape[:2] != (h, w):
            cat = cv2.resize(cat, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = (cat == HAIR_CATEGORY).astype(np.float32)

        if landmarks is not None:
            mask = _clip_beard(mask, landmarks, h)

        return mask