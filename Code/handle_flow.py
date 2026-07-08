import logging
import threading
import time
import cv2
from utils import print_text, _fit_image
from config import button_to_index, ORIGINAL_KEY, IMAGE_PATHS, SCREEN_W, DIFF_PATHS, SCREEN_H, FILTER_END, FILTER_START, COUNTDOWN
from hardware import HardwareManager
from display import DisplayManager
from face_enhance import FaceEnhancer
from alignment_guide import AlignmentGuide
from helper_functions import _apply_filters, prepare_frame
from wrinkles import CombinedWrinkleDrawer

def show_changed_grid(ctx, text, position, font_scale=1):
    canvases = list(ctx["grid_clean"])
    cell_keys = ctx["cell_keys"]
    result = []
    for i, canvas in enumerate(canvases):
        c = canvas.copy()
        key = cell_keys[i]
        if key in DIFF_PATHS:
            img = cv2.imread(DIFF_PATHS[key])
            if img is not None:
                img = _fit_image(img, SCREEN_W, SCREEN_H)
                c[:] = img
        result.append(c)
    for i, item in enumerate(result):
        result[i] = print_text(item, text, font_scale=font_scale, position=position)
    return tuple(result)


def _handle_confirm_wait(just_pressed, ctx):
    if time.time() - ctx["confirm_time"] >= 1.0:
        return "reveal"
    return "confirm_wait"


logging.basicConfig(level=logging.INFO)

class HandleFlow:
    """Main application controller."""

    def __init__(self, hardware: HardwareManager, display: DisplayManager):
        logging.info("Initializing Facial Processing Application")

        self.hardware = hardware
        self.display = display
        self.face_enhancer = FaceEnhancer()
        self.alignment_guide = AlignmentGuide()
        self.wrinkles = CombinedWrinkleDrawer()

        self.aligned_since: float | None = None
        self.text = None


    # ------------------------------------------------------------------ #
    #  HELPERS
    # ------------------------------------------------------------------ #

    def _show_cell(self, grid_frame, index, color, tint=False):
        canvases = [f.copy() for f in grid_frame]

        if tint:
            self.display.tint_cell(canvases[index], color=color, alpha=0.35)
        self.display.draw_cell_border(canvases[index], color=color)
        if self.text:
            canvases[index] = print_text(canvases[index], self.text, font_scale=1, position="top")

        self.display.update_frame(tuple(canvases), flip=False)
        return tuple(canvases)

    def _capture_and_enhance(self, frame=None):

        self.display.show_loading("Bild wird vorbereitet...", 0.0)

        if frame is None:
            frame = self.hardware.get_frame()
        if frame is None:
            logging.error("No frame to capture")
            return None

        frame = prepare_frame(frame)
        self.display.show_loading("Gesicht wird erkannt...", 0.2)

        # Force full 468-point detection
        full_landmarks = self.face_enhancer.force_detect_full(frame)
        if full_landmarks is None:
            logging.error("No face detected in photo")
            self.display.show_loading("Kein Gesicht erkannt!", progress=None)  # static error
            time.sleep(2)
            return None

        cv2.imwrite(IMAGE_PATHS["Original"], frame)

        def _filter_progress(index, total):
            progress = FILTER_START + (index / total) * (FILTER_END - FILTER_START)
            if index < total:
                self.display.show_loading(f"Filter {index + 1} wird angewendet...", progress)
            else:
                self.display.show_loading("Filter angewendet...", progress)

        self.display.show_loading("Bearbeitung wird angewendet...", FILTER_START)
        result = _apply_filters(
            self.face_enhancer, self.wrinkles, frame, full_landmarks,
            progress_callback=_filter_progress
        )
        self.display.show_loading("Abschluss...", 0.9)

        time.sleep(0.2)
        self.display.show_loading("Fertig", 1.0)
        time.sleep(0.3)  # let user see completion

        return result

    def _build_grid(self, used_keys):
        for key in IMAGE_PATHS:
            self.display.invalidate_cache(key)
        grid = self.display.make_image_grid(used_keys)
        return grid

    # ------------------------------------------------------------------ #
    #  STATE HANDLERS
    # ------------------------------------------------------------------ #

    def handle_live(self, just_pressed, ctx):
        frame = self.hardware.get_frame()
        if frame is None:
            return "live"

        raw = frame.copy()

        frame = cv2.flip(frame, 1)

        # Use FAST detection (5 points, frame-skipped)
        landmarks = self.face_enhancer.detect_landmarks_fast(frame)

        # Update alignment guide
        self.alignment_guide.update(landmarks, frame=frame)

        # Draw overlay
        frame = self.alignment_guide.draw(frame, landmarks)

        aligned = self.alignment_guide.is_aligned()

        if aligned:
            if self.aligned_since is None:
                self.aligned_since = time.time()
        else:
            self.aligned_since = None

        countdown_time = self.aligned_since is not None and time.time() - self.aligned_since >= COUNTDOWN

        if countdown_time:
            self.alignment_guide.trigger_flash()
            frame = self.alignment_guide.draw(frame, landmarks)
            self.display.update_frame(frame, flip=False)
            time.sleep(0.15)

            self.aligned_since = None
            ctx["snapshot"] = raw
            ctx["used_keys"] = None
            ctx["processing_done"] = False
            threading.Thread(target=self._run_processing, args=(ctx,), daemon=True).start()
            return "processing"

        self.display.update_frame(frame, flip=False)

        return "live"

    def handle_processing(self, just_pressed, ctx):
        if not ctx["processing_done"]:
            return "processing"

        used_keys = ctx["used_keys"]
        if not used_keys:
            self.display.show_loading("Kein Gesicht erkannt.")
            time.sleep(2.0)
            return "live"

        grid = self._build_grid(used_keys)
        ctx["cell_keys"] = self.display.cell_keys[:]  # snapshot order RIGHT NOW, before anything can reshuffle
        self.text = "In welchem Bild erkennst du dich wieder?"
        ctx["grid_clean"] = tuple(c.copy() for c in grid)
        grid_list = list(grid)
        for idx in range(len(grid_list)):
            grid_list[idx] = print_text(grid_list[idx], self.text, font_scale=0.7, position="top")
        self.display.update_frame(tuple(grid_list), flip=False)
        ctx["selected_button"] = None
        return "grid_select"

    def handle_grid_confirm(self, just_pressed, ctx):
        if not just_pressed:
            return "grid_select"

        button = just_pressed[0]
        index = button_to_index.get(button)

        # Select immediately - no need for second press
        selected = self.display.handle_button_press_with_keys(button, ctx["cell_keys"])
        is_correct = (selected == ORIGINAL_KEY)

        frame_list = [f.copy() for f in ctx["grid_clean"]]

        def _apply_tint_and_border(idx, color):
            self.display.tint_cell(frame_list[idx], color=color, alpha=0.35)

        if is_correct:
            _apply_tint_and_border(index, (0, 255, 0))
            self.text = "Das bist wirklich du!"
            text_index = index
        else:
            _apply_tint_and_border(index, (0, 0, 255))
            original_index = ctx["cell_keys"].index(ORIGINAL_KEY)  # use snapshot, not live cell_keys
            _apply_tint_and_border(original_index, (0, 255, 0))
            text_index = original_index
            self.text = "Das ist das echte Bild!"

        frame_list[text_index] = print_text(frame_list[text_index], self.text, font_scale=1, position="top")
        self.display.update_frame(tuple(frame_list), flip=False)
        ctx["confirmed_key"] = selected
        ctx["confirmed_index"] = index
        ctx["is_correct"] = is_correct
        ctx["confirm_time"] = time.time()
        return "confirm_wait"

    def show_changed_grid(self, ctx, text, position):
        canvas1 = ctx["grid_clean"][0].copy()
        canvas2 = ctx["grid_clean"][1].copy()

        for idx, key in enumerate(self.display.cell_keys):
            if key in DIFF_PATHS:
                img = cv2.imread(DIFF_PATHS[key])
                if img is not None:
                    img = _fit_image(img, self.display.cell_width, self.display.cell_height)
                    x1, y1, x2, y2 = self.display._cell_rect(idx)
                    if idx < 2:
                        canvas1[y1:y2, x1:x2] = img
                    else:
                        canvas2[y1:y2, x1:x2] = img

        canvas1 = print_text(canvas1, text, font_scale=1, position=position, style="pill")
        return canvas1, canvas2

    def handle_reveal(self, just_pressed, ctx):
        now = time.time()

        if ctx.get("reveal_start") is None:
            ctx["reveal_start"] = now
            ctx["prompt_shown"] = False
            canvas = show_changed_grid(ctx, "Hier ist was sich verändert hat.", "top")
            self.display.update_frame(canvas, flip=False)

        if not ctx["prompt_shown"] and now - ctx["reveal_start"] >= 10:
            canvas = show_changed_grid(ctx, "Drücke einen Knopf, um fortzufahren.", "bottom", 1)
            self.display.update_frame(canvas, flip=False)
            ctx["prompt_shown"] = True

        if just_pressed or (ctx["prompt_shown"] and now - ctx["reveal_start"] >= 30):
            ctx["reveal_start"] = None
            ctx["prompt_shown"] = False
            self.aligned_since = None
            self.display.reset_monitors = True
            self.alignment_guide.reset()
            return "live"

        return "reveal"

    def _run_processing(self, ctx):
        used_keys = self._capture_and_enhance(ctx.get("snapshot"))
        ctx["used_keys"] = used_keys
        ctx["processing_done"] = True