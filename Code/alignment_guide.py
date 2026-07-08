import cv2
import numpy as np
from hair_detection import Hair
from config import YAW_THRESHOLD, STABLE_FRAMES_REQUIRED, COLOR_GOOD, COLOR_BAD, COLOR_NO_FACE, OVAL_THICKNESS, \
    OVAL_RX_RATIO, OVAL_RY_RATIO, FONT_THICKNESS, FONT_SCALE_MSG, OVAL_CENTER_Y_RATIO, OVAL_FILL_ALPHA, LEFT_EYE_COLOR, \
    RIGHT_EYE_COLOR, NOSE_COLOR, CHEEK_COLOR, LANDMARK_RADIUS, EXPECTED_EYE_CHEEK_RATIO, HAIR_THRESHOLD, \
    APPLY_HAIR_DETECTION
from landmarks import NOSE_TIP_AREA, LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR, FACE_OVAL

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _draw_landmark(out, landmarks, idx, label, color, offset=10):
    """Helper to draw a single landmark with label."""
    if idx < len(landmarks):
        x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
        cv2.circle(out, (x, y), LANDMARK_RADIUS, color, -1, cv2.LINE_AA)
        cv2.putText(out, label, (x + offset, y), FONT, 0.5, color, 1, cv2.LINE_AA)


def draw_landmark_dots(out, landmarks):
    _draw_landmark(out, landmarks, LEFT_EYE_CONTOUR[8], "L_EYE", LEFT_EYE_COLOR)
    _draw_landmark(out, landmarks, RIGHT_EYE_CONTOUR[0], "R_EYE", RIGHT_EYE_COLOR)
    _draw_landmark(out, landmarks, NOSE_TIP_AREA[0], "NOSE", NOSE_COLOR)
    _draw_landmark(out, landmarks, FACE_OVAL[28], "L_CHEEK", CHEEK_COLOR)
    _draw_landmark(out, landmarks, FACE_OVAL[8], "R_CHEEK", CHEEK_COLOR)


def _is_face_in_oval(landmarks, cx: int, cy: int, rx: int, ry: int) -> bool:
    """Return True if face landmarks are inside the guide ellipse."""
    inv_rx2 = 1.0 / (rx * rx)
    inv_ry2 = 1.0 / (ry * ry)

    # For 5-point landmarks, just check the cheek points
    if len(landmarks) == 5:
        for i in range(3, 5):  # Left and right cheek
            x, y = landmarks[i][0], landmarks[i][1]
            dx = x - cx
            dy = y - cy
            if (dx * dx * inv_rx2) + (dy * dy * inv_ry2) > 1.0:
                return False
        return True

    # Full 468-point landmarks
    for idx in FACE_OVAL:
        if idx >= len(landmarks):
            return False
        x, y = landmarks[idx][0], landmarks[idx][1]
        dx = x - cx
        dy = y - cy
        if (dx * dx * inv_rx2) + (dy * dy * inv_ry2) > 1.0:
            return False
    return True


class AlignmentGuide:
    """Stateful overlay that tracks head alignment across frames."""

    def __init__(self):
        self.current_yaw: float = 0.0
        self.current_face_visible: bool = False
        self._stable_count: int = 0
        self.draw_landmarks: bool = False  # Set to True to see landmarks
        self.face_in_oval: bool = False
        self._cached_frame_size = None
        self._cached_ellipse_params = None
        self._oval_fill_cache: dict = {}
        self._text_cache: dict = {}

        self.flash_intensity = 0.0
        self.flash_active = False
        self.flash_decay_rate = 0.05

        self.draw_hair_mask: bool = False

        self.hair_threshold = HAIR_THRESHOLD

        self.hair = Hair() if APPLY_HAIR_DETECTION else None
        self.hair_ratio = 0.0
        self._hair_frame_counter = 0
        self._hair_update_interval = 5  # update every 5 frames
        self.apply_hair_detection = APPLY_HAIR_DETECTION

    def _get_face_roi_mask(self, w, h, landmarks=None) -> np.ndarray:
        cx, cy, rx, ry = self._get_ellipse_params(w, h)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1, -1)

        # Use nose tip landmark for a precise cutoff if available
        if landmarks is not None and len(landmarks) >= 3:
            nose_idx = 2 if len(landmarks) == 5 else 1  # 5-pt vs full 468-pt
            cutoff_y = int(landmarks[nose_idx][1])
        else:
            # fallback: old fixed estimate
            cutoff_y = int(cy - ry * 0.1)

        mask[cutoff_y:, :] = 0
        return mask

    def update(self, landmarks, frame=None, frame_shape=None) -> None:
        """Update with landmarks and optional frame for hair detection."""
        if landmarks is None or len(landmarks) == 0:
            self.current_face_visible = False
            self.current_yaw = 1.0
            self._stable_count = 0
            self.face_in_oval = False
            self.hair_ratio = 0.0
            return

        self.current_face_visible = True

        if self._cached_frame_size is None:
            if frame_shape is None and frame is not None:
                frame_shape = frame.shape
            if frame_shape is None:
                self.current_face_visible = False
                return
            h, w = frame_shape[:2]
            self._get_ellipse_params(w, h)

        w, h = self._cached_frame_size
        cx, cy, rx, ry = self._cached_ellipse_params

        self.face_in_oval = _is_face_in_oval(landmarks, cx, cy, rx, ry)

        if self.face_in_oval:
            if len(landmarks) == 5:
                self.current_yaw = self._compute_yaw_from_5_points(landmarks)
            else:
                self.current_yaw = self._compute_yaw(landmarks)
        else:
            self.current_yaw = 1.0

        if self.face_in_oval and self.current_yaw <= YAW_THRESHOLD:
            self._stable_count = min(self._stable_count + 1, STABLE_FRAMES_REQUIRED)
        else:
            self._stable_count = 0

        # STEP 4: Compute hair ratio only when face is in oval and we have a frame
        if (
                self.apply_hair_detection
                and self.hair is not None
                and self.face_in_oval
                and frame is not None
        ):
            self._hair_frame_counter += 1
            if self._hair_frame_counter >= self._hair_update_interval:
                self._hair_frame_counter = 0
                face_mask = self._get_face_roi_mask(w, h, landmarks=landmarks)
                self.hair_ratio = self.hair.get_hair_ratio(frame, face_mask)
        else:
            self.hair_ratio = 0.0

    def is_aligned(self) -> bool:
        """Return True only when face is straight, inside oval, and hair coverage is low."""
        hair_ok = (
            self.hair_ratio <= self.hair_threshold
            if self.apply_hair_detection
            else True
        )

        return (
                self.current_face_visible
                and self.current_yaw <= YAW_THRESHOLD
                and self._stable_count >= STABLE_FRAMES_REQUIRED
                and self.face_in_oval
                and hair_ok
        )

    def trigger_flash(self):
        """Trigger a flash effect."""
        self.flash_intensity = 1.0
        self.flash_active = True

    def update_flash(self):
        """Decay flash intensity each frame."""
        if self.flash_active:
            self.flash_intensity -= self.flash_decay_rate
            if self.flash_intensity <= 0:
                self.flash_intensity = 0
                self.flash_active = False

    def draw(self, frame: np.ndarray, landmarks=None) -> np.ndarray:
        """Return a new frame with landmarks drawn."""
        out = frame.copy() if frame is not None else None
        if out is None:
            return frame

        h, w = out.shape[:2]
        cx, cy, rx, ry = self._get_ellipse_params(w, h)

        # Draw landmarks if available and enabled
        if landmarks is not None and len(landmarks) > 0 and self.current_face_visible and self.draw_landmarks:
            draw_landmark_dots(out, landmarks)

        if self.current_face_visible:
            color = COLOR_GOOD if self.is_aligned() else COLOR_BAD
        else:
            color = COLOR_NO_FACE


        cv2.ellipse(out, (cx, cy), (rx, ry), 0, 0, 360, color, OVAL_THICKNESS,
                    lineType=cv2.LINE_AA)

        if self.draw_hair_mask and self.hair and frame is not None and self.face_in_oval:
            out = self.hair.apply_mask_on_image(out)

        # Status message
        self._draw_status_msg(out, w, h, color)

        return out

    def _get_ellipse_params(self, w, h):
        if (w, h) != self._cached_frame_size:
            self._cached_frame_size = (w, h)
            ref = min(w, h)  # use the shorter axis as reference
            self._cached_ellipse_params = (
                w // 2,
                int(h * OVAL_CENTER_Y_RATIO),
                int(ref * OVAL_RX_RATIO),
                int(ref * OVAL_RY_RATIO)
            )
        return self._cached_ellipse_params

    def _yaw_from_points(self, left_eye, right_eye, nose_tip, left_cheek, right_cheek) -> float:
        eye_dist = np.linalg.norm(right_eye - left_eye)
        if eye_dist < 1e-3:
            return 1.0
        eye_mid = (left_eye + right_eye) / 2.0
        signal_a = abs(nose_tip[0] - eye_mid[0]) / (eye_dist * 0.5)
        d_left = np.linalg.norm(nose_tip - left_cheek)
        d_right = np.linalg.norm(nose_tip - right_cheek)
        total = d_left + d_right
        signal_b = abs(d_left - d_right) / total * 2.0 if total > 1e-3 else 0.0
        cheek_dist = np.linalg.norm(right_cheek - left_cheek)
        signal_c = max(0.0, (
                    EXPECTED_EYE_CHEEK_RATIO - eye_dist / cheek_dist) / EXPECTED_EYE_CHEEK_RATIO) if cheek_dist > 1e-3 else 0.0
        return float(np.clip(0.40 * signal_a + 0.35 * signal_b + 0.25 * signal_c, 0.0, 1.0))

    def _compute_yaw_from_5_points(self, lm) -> float:
        pts = [np.array(lm[i], dtype=np.float64) for i in range(5)]
        return self._yaw_from_points(*pts)

    def _compute_yaw(self, lm) -> float:
        idxs = [LEFT_EYE_CONTOUR[8], RIGHT_EYE_CONTOUR[0], NOSE_TIP_AREA[0], FACE_OVAL[28], FACE_OVAL[8]]
        if not all(i < len(lm) for i in idxs):
            return 1.0
        pts = [np.array([lm[i][0], lm[i][1]], dtype=np.float64) for i in idxs]
        return self._yaw_from_points(*pts)

    def _get_status_message(self) -> str:
        if not self.current_face_visible:
            return "Kein Gesicht wurde gefunden"
        if not self.face_in_oval:
            return "Bitte bewege dein Gesicht in den Kreis"
        if self.current_yaw > YAW_THRESHOLD:
            return "Bitte schaue in die Kamera"
        if self.hair_ratio > self.hair_threshold and self.apply_hair_detection:
            return "Bitte nimm deine Haare aus dem Gesicht"
        if self.is_aligned():
            return "Halte still und warte"
        return "Halte still..."

    def _draw_status_msg(self, frame: np.ndarray, w: int, h: int, color) -> None:
        """Status text just above the yaw bar."""
        msg = self._get_status_message()

        # Cache text size calculation
        cache_key = (msg, w, h)

        if cache_key not in self._text_cache:
            (tw, th), _ = cv2.getTextSize(msg, FONT, FONT_SCALE_MSG, FONT_THICKNESS)
            self._text_cache[cache_key] = (tw, th)
        else:
            tw, th = self._text_cache[cache_key]

        tx = (w - tw) // 2
        ty = int(h * OVAL_CENTER_Y_RATIO) - int(h * OVAL_RY_RATIO) - 20

        cv2.putText(frame, msg, (tx + 1, ty + 1), FONT,
                    FONT_SCALE_MSG, (0, 0, 0), FONT_THICKNESS + 1,
                    lineType=cv2.LINE_AA)
        cv2.putText(frame, msg, (tx, ty), FONT,
                    FONT_SCALE_MSG, color, FONT_THICKNESS,
                    lineType=cv2.LINE_AA)

    def reset(self) -> None:
        """Reset alignment state for a fresh check."""
        self._stable_count = 0
        self.current_yaw = 1.0
        self.current_face_visible = False
        self.face_in_oval = False
        self._oval_fill_cache.clear()
        self._text_cache.clear()
        self.hair_ratio = 0.0
        self._hair_frame_counter = 0