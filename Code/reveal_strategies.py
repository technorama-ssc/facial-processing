import cv2
import time
from abc import ABC, abstractmethod

import numpy as np

from utils import print_text, _fit_image
from config import DIFF_PATHS, SCREEN_W, SCREEN_H, IMAGE_PATHS


class RevealStrategy(ABC):
    """Base class for reveal strategies."""

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_description(self) -> str:
        pass

    @abstractmethod
    def get_initial_grid(self, ctx) -> tuple:
        """Return the initial grid to display."""
        pass

    @abstractmethod
    def update(self, ctx, just_pressed) -> tuple:
        """Update the display. Returns (grid, should_exit, should_reset)."""
        pass


def _make_colored_grid(ctx, text="Here's what changed"):
    """Show colored diff overlays on the filtered images."""
    if "grid_clean" not in ctx or ctx["grid_clean"] is None:
        return None

    canvases = list(ctx["grid_clean"])
    cell_keys = ctx["cell_keys"]
    result = []

    for i, canvas in enumerate(canvases):
        c = canvas.copy()
        key = cell_keys[i]
        if key in DIFF_PATHS:
            diff_img = cv2.imread(DIFF_PATHS[key])
            if diff_img is not None:
                diff_img = _fit_image(diff_img, SCREEN_W, SCREEN_H)
                c[:] = diff_img
        result.append(c)

    for i, item in enumerate(result):
        result[i] = print_text(item, text, font_scale=1, position="top", style="pill")

    return tuple(result)


def _make_filtered_grid(ctx):
    """Show the final filtered images."""
    if "grid_clean" not in ctx or ctx["grid_clean"] is None:
        return None

    filtered_list = [c.copy() for c in ctx["grid_clean"]]
    for i in range(len(filtered_list)):
        filtered_list[i] = print_text(
            filtered_list[i],
            "Here's the result",
            font_scale=1.0,
            position="top",
            style="pill"
        )
    return tuple(filtered_list)


def _make_split_grid(ctx):
    """Show colored diff on left, filtered image on right."""
    if "grid_clean" not in ctx or ctx["grid_clean"] is None:
        return None

    canvases = list(ctx["grid_clean"])
    cell_keys = ctx["cell_keys"]
    result = []

    for i, canvas in enumerate(canvases):
        key = cell_keys[i]
        h, w = canvas.shape[:2]
        half_w = w // 2

        # Left half: Colored diff
        left = canvas[:, :half_w].copy()
        if key in DIFF_PATHS:
            diff_img = cv2.imread(DIFF_PATHS[key])
            if diff_img is not None:
                diff_img = _fit_image(diff_img, half_w, h)
                left = diff_img[:, :half_w].copy() if diff_img.shape[1] > half_w else diff_img

        # Right half: Filtered result
        right = canvas[:, half_w:].copy()
        filtered_img = cv2.imread(IMAGE_PATHS.get(key, ""))
        if filtered_img is not None:
            filtered_img = _fit_image(filtered_img, half_w, h)
            right = filtered_img[:, half_w:].copy() if filtered_img.shape[1] > half_w else filtered_img

        # Combine
        combined = np.hstack([left, right])

        # Add labels
        combined = print_text(combined, "Colored Diff", font_scale=0.7, position="top")
        combined = print_text(combined, "Filtered", font_scale=0.7, position="top")

        result.append(combined)

    for i, item in enumerate(result):
        result[i] = print_text(item, "Compare", font_scale=1, position="top", style="pill")

    return tuple(result)


def _make_subtle_grid(ctx):
    """Show colored overlays at 8% opacity."""
    if "grid_clean" not in ctx or ctx["grid_clean"] is None:
        return None

    canvases = list(ctx["grid_clean"])
    cell_keys = ctx["cell_keys"]
    result = []

    for i, canvas in enumerate(canvases):
        c = canvas.copy()
        key = cell_keys[i]

        if key in DIFF_PATHS:
            diff_img = cv2.imread(DIFF_PATHS[key])
            if diff_img is not None:
                diff_img = _fit_image(diff_img, SCREEN_W, SCREEN_H)
                # ✅ LOW OPACITY: 8% diff overlay
                c = cv2.addWeighted(c, 0.92, diff_img, 0.08, 0)

        result.append(c)

    for i, item in enumerate(result):
        result[i] = print_text(item, "Subtle changes", font_scale=1, position="top", style="pill")

    return tuple(result)


def _add_exit_prompt(grid, text="Press any button to continue"):
    """Add exit prompt to grid."""
    if grid is None:
        return None

    grid = list(grid)
    for i in range(len(grid)):
        grid[i] = print_text(
            grid[i],
            text,
            font_scale=0.7,
            position="bottom",
            style="pill"
        )
    return tuple(grid)


class StandardReveal(RevealStrategy):
    """Colored overlay 5s → filtered images 30s"""

    def get_name(self) -> str:
        return "Standard"

    def get_description(self) -> str:
        return "Colored overlay 5s → filtered images 30s"

    def get_initial_grid(self, ctx):
        ctx["reveal_stage"] = "colored"
        ctx["reveal_start"] = time.time()
        return _make_colored_grid(ctx)

    def update(self, ctx, just_pressed):
        now = time.time()

        if ctx.get("reveal_stage") == "colored":
            if now - ctx.get("reveal_start", now) >= 5.0:
                ctx["reveal_stage"] = "filtered"
                ctx["reveal_start"] = now
                return _make_filtered_grid(ctx), False, False
            return None, False, False

        if ctx.get("reveal_stage") == "filtered":
            if not ctx.get("prompt_shown") and now - ctx.get("reveal_start", now) >= 1.5:
                ctx["prompt_shown"] = True
                return _add_exit_prompt(_make_filtered_grid(ctx)), False, False

            if now - ctx.get("reveal_start", now) >= 30.0 or just_pressed:
                return None, True, True

            return None, False, False

        return None, False, False


class SlideshowReveal(RevealStrategy):
    """Alternates colored/filtered every 2.5 seconds for 30 seconds total"""

    def get_name(self) -> str:
        return "Slideshow"

    def get_description(self) -> str:
        return "Alternates colored/filtered every 2.5s"

    def get_initial_grid(self, ctx):
        ctx["slideshow_show_colored"] = True
        ctx["slideshow_last_switch"] = time.time()
        ctx["reveal_start"] = time.time()
        return _make_colored_grid(ctx)

    def update(self, ctx, just_pressed):
        now = time.time()

        if now - ctx.get("slideshow_last_switch", now) >= 2.5:
            ctx["slideshow_show_colored"] = not ctx.get("slideshow_show_colored", True)
            ctx["slideshow_last_switch"] = now

            if ctx["slideshow_show_colored"]:
                grid = _make_colored_grid(ctx)
            else:
                grid = _make_filtered_grid(ctx)

            if not ctx.get("prompt_shown"):
                ctx["prompt_shown"] = True
                grid = _add_exit_prompt(grid)

            return grid, False, False

        if not ctx.get("prompt_shown"):
            ctx["prompt_shown"] = True
            if ctx.get("slideshow_show_colored", True):
                grid = _make_colored_grid(ctx)
            else:
                grid = _make_filtered_grid(ctx)
            return _add_exit_prompt(grid), False, False

        if now - ctx.get("reveal_start", now) >= 30.0 or just_pressed:
            return None, True, True

        return None, False, False


class SubtleReveal(RevealStrategy):
    """All 4 colored images with low opacity overlays"""

    def get_name(self) -> str:
        return "Subtle"

    def get_description(self) -> str:
        return "Colored overlays at 8% opacity"

    def get_initial_grid(self, ctx):
        ctx["reveal_start"] = time.time()
        return _make_subtle_grid(ctx)

    def update(self, ctx, just_pressed):
        now = time.time()

        if not ctx.get("prompt_shown") and now - ctx.get("reveal_start", now) >= 1.5:
            ctx["prompt_shown"] = True
            return _add_exit_prompt(_make_subtle_grid(ctx)), False, False

        if now - ctx.get("reveal_start", now) >= 30.0 or just_pressed:
            return None, True, True

        return None, False, False


class SplitReveal(RevealStrategy):
    """Colored and filtered side by side on same screen"""

    def get_name(self) -> str:
        return "Split View"

    def get_description(self) -> str:
        return "Colored (left) + Filtered (right) side by side"

    def get_initial_grid(self, ctx):
        ctx["reveal_start"] = time.time()
        return _make_split_grid(ctx)

    def update(self, ctx, just_pressed):
        now = time.time()

        if not ctx.get("prompt_shown") and now - ctx.get("reveal_start", now) >= 1.5:
            ctx["prompt_shown"] = True
            return _add_exit_prompt(_make_split_grid(ctx)), False, False

        if now - ctx.get("reveal_start", now) >= 30.0 or just_pressed:
            return None, True, True

        return None, False, False


REVEAL_STRATEGIES = {
    "standard": StandardReveal(),
    "slideshow": SlideshowReveal(),
    "split": SplitReveal(),
    "subtle": SubtleReveal(),
}

DEFAULT_STRATEGY = "standard"
_current_strategy = DEFAULT_STRATEGY


def get_strategy() -> str:
    return _current_strategy


def set_strategy(name: str) -> bool:
    global _current_strategy
    if name in REVEAL_STRATEGIES:
        _current_strategy = name
        return True
    return False


def get_strategies() -> list:
    return [
        {"id": k, "name": v.get_name(), "description": v.get_description()}
        for k, v in REVEAL_STRATEGIES.items()
    ]


def get_current_strategy_instance() -> RevealStrategy:
    return REVEAL_STRATEGIES[_current_strategy]