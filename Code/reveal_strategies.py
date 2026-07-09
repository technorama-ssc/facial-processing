import cv2
import time
from abc import ABC, abstractmethod
from utils import print_text, _fit_image
from config import DIFF_PATHS, SCREEN_W, SCREEN_H


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
    """Helper: create colored diff overlay grid."""
    from handle_flow import show_changed_grid
    return show_changed_grid(ctx, text, "top", font_scale=1)


def _make_filtered_grid(ctx):
    """Helper: create filtered results grid."""
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


def _add_exit_prompt(grid, text="Press any button to continue"):
    """Helper: add exit prompt to grid."""
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


def _make_slideshow_grid(ctx, show_original):
    """Helper: create slideshow grid."""
    canvases = list(ctx["grid_clean"])
    cell_keys = ctx["cell_keys"]
    result = []

    for i, canvas in enumerate(canvases):
        c = canvas.copy()
        key = cell_keys[i]

        if show_original:
            c = print_text(c, "Original", font_scale=0.6, position="top")
        else:
            if key in DIFF_PATHS:
                diff_img = cv2.imread(DIFF_PATHS[key])
                if diff_img is not None:
                    diff_img = _fit_image(diff_img, SCREEN_W, SCREEN_H)
                    c[:] = diff_img
            c = print_text(c, "Filtered", font_scale=0.6, position="top")

        result.append(c)

    return tuple(result)


def _make_subtle_grid(ctx):
    """Helper: create subtle overlay grid."""
    from handle_flow import show_changed_grid
    return show_changed_grid(ctx, "Subtle changes", "top", font_scale=1, variant="subtle")


def _make_split_grid(ctx):
    """Helper: create split view grid."""
    from handle_flow import show_changed_grid
    return show_changed_grid(ctx, "Compare", "top", font_scale=1, variant="split")


class StandardReveal(RevealStrategy):
    """Original behavior: colored overlay, then filtered images."""

    def get_name(self) -> str:
        return "Standard"

    def get_description(self) -> str:
        return "Colored overlay 5s → filtered images"

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
    """Alternates between original and filtered every 2 seconds."""

    def get_name(self) -> str:
        return "Slideshow"

    def get_description(self) -> str:
        return "Alternates original/filtered every 2s"

    def get_initial_grid(self, ctx):
        ctx["slideshow_show_original"] = True
        ctx["slideshow_last_switch"] = time.time()
        ctx["reveal_start"] = time.time()
        return _make_slideshow_grid(ctx, True)

    def update(self, ctx, just_pressed):
        now = time.time()

        if now - ctx.get("slideshow_last_switch", now) >= 2.0:
            ctx["slideshow_show_original"] = not ctx.get("slideshow_show_original", True)
            ctx["slideshow_last_switch"] = now
            grid = _make_slideshow_grid(ctx, ctx["slideshow_show_original"])

            if not ctx.get("prompt_shown") and now - ctx.get("reveal_start", now) >= 1.5:
                ctx["prompt_shown"] = True
                grid = _add_exit_prompt(grid)

            return grid, False, False

        if not ctx.get("prompt_shown") and now - ctx.get("reveal_start", now) >= 1.5:
            ctx["prompt_shown"] = True
            grid = _make_slideshow_grid(ctx, ctx.get("slideshow_show_original", True))
            return _add_exit_prompt(grid), False, False

        if now - ctx.get("reveal_start", now) >= 30.0 or just_pressed:
            return None, True, True

        return None, False, False


class SubtleReveal(RevealStrategy):
    """Shows colored areas at very low opacity."""

    def get_name(self) -> str:
        return "Subtle"

    def get_description(self) -> str:
        return "Colored overlay at 8% opacity"

    def get_initial_grid(self, ctx):
        ctx["reveal_start"] = time.time()
        return _make_subtle_grid(ctx)

    def update(self, ctx, just_pressed):
        now = time.time()

        if not ctx.get("prompt_shown") and now - ctx.get("reveal_start", now) >= 1.5:
            ctx["prompt_shown"] = True
            return _add_exit_prompt(_make_subtle_grid(ctx)), False, False

        if now - ctx.get("reveal_start", now) >= 5.0 or just_pressed:
            return None, True, True

        return None, False, False


class SplitReveal(RevealStrategy):
    """Side-by-side comparison."""

    def get_name(self) -> str:
        return "Split View"

    def get_description(self) -> str:
        return "Left: original, Right: filtered"

    def get_initial_grid(self, ctx):
        ctx["reveal_start"] = time.time()
        return _make_split_grid(ctx)

    def update(self, ctx, just_pressed):
        now = time.time()

        if not ctx.get("prompt_shown") and now - ctx.get("reveal_start", now) >= 1.5:
            ctx["prompt_shown"] = True
            return _add_exit_prompt(_make_split_grid(ctx)), False, False

        if now - ctx.get("reveal_start", now) >= 5.0 or just_pressed:
            return None, True, True

        return None, False, False


REVEAL_STRATEGIES = {
    "standard": StandardReveal(),
    "slideshow": SlideshowReveal(),
    "subtle": SubtleReveal(),
    "split": SplitReveal(),
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