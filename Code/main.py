import logging
import threading
import time
import os
from config import IMAGE_PATHS, DIFF_PATHS
from hardware import HardwareManager
from display import DisplayManager
from face_enhance import FaceEnhancer
from alignment_guide import AlignmentGuide
from handle_flow import HandleFlow, _handle_confirm_wait
from webserver import start_webserver
from wrinkles import CombinedWrinkleDrawer

logging.basicConfig(level=logging.WARNING)


class FacialProcessingApp:
    """Main application controller."""

    def __init__(self):
        logging.info("Initializing Facial Processing Application")

        self.hardware = HardwareManager()
        self.display = DisplayManager(self.hardware)
        self.face_enhancer = FaceEnhancer()
        self.alignment_guide = AlignmentGuide()
        self.wrinkles = CombinedWrinkleDrawer()

        self.flow = HandleFlow(self.hardware, self.display)

        self.aligned_since: float | None = None
        self.text = None
        self.frame_count = 0

        start_webserver(self.display, self.hardware, port=5000, app_ref=self)

    def run(self):
        """Start the application."""
        try:
            logging.info("Starting application")
            threading.Thread(target=self._feed_frames, daemon=True).start()
            self.display.run()
        finally:
            self.cleanup()


    # ------------------------------------------------------------------ #
    #  MAIN LOOP
    # ------------------------------------------------------------------ #

    def _feed_frames(self):
        state = "live"
        ctx = {"grid": None, "selected_button": None}
        prev_state = state

        # Frame rate control
        target_fps = 30
        frame_time = 1.0 / target_fps
        last_frame_time = time.time()

        handlers = {
            "live": self.flow.handle_live,
            "processing": self.flow.handle_processing,
            "grid_select": self.flow.handle_grid_confirm,
            "confirm_wait": _handle_confirm_wait,
            "reveal": self.flow.handle_reveal,
        }

        while self.hardware.running:
            current_time = time.time()
            elapsed = current_time - last_frame_time

            # Rate limiting
            if elapsed < frame_time:
                time.sleep(max(0, frame_time - elapsed - 0.001))
                continue

            just_pressed = self.hardware.check_buttons()

            if state != prev_state:
                just_pressed = []
                prev_state = state

            new_state = handlers[state](just_pressed, ctx)

            if new_state != state:
                prev_state = state
                state = new_state
                self.hardware.check_buttons()

            last_frame_time = time.time()

    # ------------------------------------------------------------------ #

    def cleanup(self):
        logging.info("Shutting down application")
        try:
            self.hardware.cleanup()
        except Exception as e:
            logging.error(f"Error cleaning up hardware: {e}")

        all_paths = list(IMAGE_PATHS.values()) + list(DIFF_PATHS.values())
        for path in all_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    logging.info(f"Deleted {path}")
            except Exception as e:
                logging.error(f"Could not delete {path}: {e}")


def main():
    app = FacialProcessingApp()
    app.run()


if __name__ == "__main__":
    main()