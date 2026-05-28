import os
import cv2
import numpy as np
import threading
from utils import random_list, print_text, _fit_image
from config import (
    IMAGE_PATHS, button_to_index, SCREEN_H, SCREEN_W, MONITOR_POSITIONS
)
import time


class DisplayManager:

    def __init__(self, hardware):
        self.running = True
        self.hardware = hardware
        self.cv_window_names = [f"Camera Feed {i + 1}" for i in range(4)]

        time.sleep(3)

        for i, (x, y) in enumerate(MONITOR_POSITIONS):
            cv2.namedWindow(self.cv_window_names[i], cv2.WINDOW_NORMAL)
            cv2.moveWindow(self.cv_window_names[i], x + 1, y + 1)  # +1 forces Wayland to register position
            cv2.resizeWindow(self.cv_window_names[i], SCREEN_W, SCREEN_H)
            cv2.waitKey(100)  # give Wayland time to process each window
            cv2.moveWindow(self.cv_window_names[i], x, y)
            cv2.setWindowProperty(self.cv_window_names[i], cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.waitKey(100)

        self.waiting_frame = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
        self.waiting_frame = print_text(self.waiting_frame, "Warten...", font_scale=2, position="center",
                                        style="outline")
        self.reset_monitors = False

        self.display_frame = None
        self.preloaded_grid_images = {}

        self.frame_lock = threading.Lock()

        self.prev_cell_keys = None

        self.cell_width = SCREEN_W
        self.cell_height = SCREEN_H

        keys = list(IMAGE_PATHS.keys())
        self.cell_keys = random_list(keys)



    def invalidate_cache(self, key):
        """Remove a cached image so it will be reloaded from disk on next access."""
        self.preloaded_grid_images.pop(key, None)

    def get_grid_image(self, key):
        """Return image scaled to fit the computed grid cell size."""
        if key not in self.preloaded_grid_images:
            path = IMAGE_PATHS.get(key)
            if path and os.path.exists(path):
                img = cv2.imread(path)
                if img is not None:
                    img = _fit_image(img, self.cell_width, self.cell_height)
                    self.preloaded_grid_images[key] = img
        return self.preloaded_grid_images.get(key)

    def update_frame(self, frame, flip):
        """Update display with processed frame - NO COPY."""
        if frame is not None:
            with self.frame_lock:
                self.display_frame = cv2.flip(frame, 1) if flip else frame

    def draw_loading_bar(self, frame, text, progress, bar_width=400, bar_height=30):
        """
        Draw a loading bar and text on a single frame.
        progress: float between 0.0 and 1.0
        Returns the modified frame.
        """
        h, w = frame.shape[:2]
        # Background (semi-transparent black overlay)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # Bar coordinates (centered)
        bar_x = (w - bar_width) // 2
        bar_y = (h - bar_height) // 2
        # Outer border
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                      (200, 200, 200), 2)
        # Filled portion
        filled = int(bar_width * progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_height),
                      (0, 255, 0), -1)

        # Percentage text
        percent_text = f"{int(progress * 100)}%"
        (tw, th), _ = cv2.getTextSize(percent_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        tx = bar_x + (bar_width - tw) // 2
        ty = bar_y - 10
        cv2.putText(frame, percent_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)

        # Main text above the bar
        (tw2, th2), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        tx2 = (w - tw2) // 2
        ty2 = bar_y - 40
        cv2.putText(frame, text, (tx2, ty2), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2)

        return frame

    def show_loading(self, text="Processing...", progress=None):
        """Show loading screen. If progress (0..1) is given, draw a bar."""
        blank = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
        if progress is not None:
            frame = self.draw_loading_bar(blank, text, progress)
        else:
            frame = print_text(blank, text, font_scale=1, position="center", style="outline")
        self.update_frame(tuple([frame] * 4), flip=False)

    def make_image_grid(self, keys, original_key="Original"):
        """
        Returns a tuple of 4 full‑screen canvases.
        Ensures that 'original_key' is not placed at the same index as in the previous call.
        """
        keys = list(keys)
        while len(keys) < 4:
            keys.append(original_key)

        max_attempts = 50
        shuffled = None
        for _ in range(max_attempts):
            shuffled = random_list(keys)
            if self.prev_cell_keys is not None:
                conflict = False
                for idx in range(len(shuffled)):
                    if shuffled[idx] == original_key and self.prev_cell_keys[idx] == original_key:
                        conflict = True
                        break
                if conflict:
                    continue

            self.cell_keys = shuffled.copy()
            self.prev_cell_keys = shuffled.copy()
            break
        else:
            self.cell_keys = shuffled.copy()
            self.prev_cell_keys = shuffled.copy()

        canvases = []
        for i in range(4):
            canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
            img = self.get_grid_image(self.cell_keys[i])
            if img is None:
                img = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
            else:
                img = _fit_image(img, SCREEN_W, SCREEN_H)
            canvas[:] = img
            canvases.append(canvas)

        return tuple(canvases)


    def handle_button_press(self, button):
        """Map physical button press to the image in the grid and check correctness."""
        index = button_to_index.get(button)  # Button → grid index
        if index is not None and index < len(self.cell_keys):
            selected_image_key = self.cell_keys[index]
            print(f"Button {button} selected image: {selected_image_key}")

            return selected_image_key
        return None

    def handle_button_press_with_keys(self, button, cell_keys):
        """Like handle_button_press but uses a caller-supplied key snapshot.

        Always use this version after grid is built so a reshuffle can't
        corrupt the mapping between what's on screen and what a button means.
        """
        index = button_to_index.get(button)
        if index is not None and index < len(cell_keys):
            selected_image_key = cell_keys[index]
            print(f"Button {button} selected image: {selected_image_key}")
            return selected_image_key
        return None

    def _cell_rect(self):
        return 0, 0, self.cell_width, self.cell_height

    def draw_cell_border(self, canvas, color=(0, 255, 0), thickness=40):
        cv2.rectangle(canvas, (0, 0), (self.cell_width - 1, self.cell_height - 1), color, thickness)

    def tint_cell(self, canvas, color, alpha=0.35):
        overlay = np.full_like(canvas, color, dtype=np.uint8)
        np.copyto(canvas, cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0))

    def run(self):
        for i in range(1, 4):  # show waiting on monitors 2,3,4
            cv2.imshow(self.cv_window_names[i], self.waiting_frame)

        last_frame = None

        while self.running:
            with self.frame_lock:
                frame = self.display_frame

            if frame is not None:
                if isinstance(frame, tuple):
                    for i, f in enumerate(frame):
                        cv2.imshow(self.cv_window_names[i], f)
                else:
                    if self.reset_monitors:
                        for i in range(1, 4):
                            cv2.imshow(self.cv_window_names[i], self.waiting_frame)
                        self.reset_monitors = False

                    if not np.array_equal(frame, last_frame):
                        cv2.imshow(self.cv_window_names[0], frame)
                        last_frame = frame.copy()

            cv2.waitKey(10)