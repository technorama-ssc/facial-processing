import cv2
import time
from abc import ABC, abstractmethod

import config as cfg
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


def _add_green_border_to_original(grid, ctx):
    """Add a green border around the original image ONCE."""
    if grid is None or "cell_keys" not in ctx:
        return grid

    # Check if we already have a cached version with border
    if ctx.get("_bordered_grid") is not None:
        return ctx["_bordered_grid"]

    grid_list = list(grid)
    cell_keys = ctx["cell_keys"]

    for i, key in enumerate(cell_keys):
        if key == "Original":
            # Draw a thick green border around the original image
            cv2.rectangle(grid_list[i], (8, 8),
                          (SCREEN_W - 8, SCREEN_H - 8),
                          (0, 255, 0), 10)
            # Add a subtle glow effect with a second thinner border
            cv2.rectangle(grid_list[i], (18, 18),
                          (SCREEN_W - 18, SCREEN_H - 18),
                          (100, 255, 100), 3)
            break

    bordered = tuple(grid_list)
    ctx["_bordered_grid"] = bordered  # Cache it!
    return bordered


def _add_exit_prompt(grid, text="Drücke einen Knopf um fortzufahren."):
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


def _make_colored_grid(ctx):
    """Show colored diff overlays on the filtered images."""
    if "grid_clean" not in ctx or ctx["grid_clean"] is None:
        return None

    # Check cache first
    cache_key = "colored_grid"
    if ctx.get(cache_key) is not None:
        return ctx[cache_key]

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

    grid = tuple(result)
    ctx[cache_key] = grid  # Cache it
    return grid


def _make_filtered_grid(ctx):
    """Show the final filtered images."""
    if "grid_clean" not in ctx or ctx["grid_clean"] is None:
        return None

    # Check cache first
    cache_key = "filtered_grid"
    if ctx.get(cache_key) is not None:
        return ctx[cache_key]

    filtered_list = [c.copy() for c in ctx["grid_clean"]]
    grid = tuple(filtered_list)
    ctx[cache_key] = grid  # Cache it
    return grid


def _make_dissolve_grid(ctx, alpha):
    """Blend colored diff (alpha=0) into filtered result (alpha=1)."""
    if "grid_clean" not in ctx or ctx["grid_clean"] is None:
        return None

    canvases = list(ctx["grid_clean"])
    cell_keys = ctx["cell_keys"]
    result = []

    for i, canvas in enumerate(canvases):
        key = cell_keys[i]

        diff_img = cv2.imread(DIFF_PATHS.get(key, ""))
        filtered_img = cv2.imread(IMAGE_PATHS.get(key, ""))

        if diff_img is not None:
            diff_img = _fit_image(diff_img, SCREEN_W, SCREEN_H)
        else:
            diff_img = canvas.copy()

        if filtered_img is not None:
            filtered_img = _fit_image(filtered_img, SCREEN_W, SCREEN_H)
        else:
            filtered_img = canvas.copy()

        blended = cv2.addWeighted(diff_img, 1 - alpha, filtered_img, alpha, 0)
        result.append(blended)

    return tuple(result)


def _make_subtle_grid(ctx):
    """Show colored overlays at 50% opacity."""
    if "grid_clean" not in ctx or ctx["grid_clean"] is None:
        return None

    # Check cache first
    cache_key = "subtle_grid"
    if ctx.get(cache_key) is not None:
        return ctx[cache_key]

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
                c = cv2.addWeighted(c, 0.50, diff_img, 0.50, 0)

        result.append(c)

    grid = tuple(result)
    ctx[cache_key] = grid  # Cache it
    return grid


def _prepare_reveal_grid(grid, ctx, prompt_text="Drücke einen Knopf um fortzufahren."):
    """
    Prepare a grid for display by:
    1. Adding green border to original (cached)
    2. Adding exit prompt
    Returns the final grid ready for display.
    """
    if grid is None:
        return None

    # Add green border (uses cache internally)
    grid_with_border = _add_green_border_to_original(grid, ctx)

    # Add exit prompt (always re-applied since it has text overlay)
    grid_with_prompt = _add_exit_prompt(grid_with_border, prompt_text)

    return grid_with_prompt


class SlideshowReveal(RevealStrategy):
    """Alternates colored/filtered every 2.5s, crossfading smoothly between them"""

    HOLD = 2.0
    TRANSITION = 1.5
    TOTAL = 60.0

    def get_name(self) -> str:
        return "Slideshow"

    def get_description(self) -> str:
        return "Alternates colored/filtered every 2.5s with a smooth crossfade"

    def get_initial_grid(self, ctx):
        ctx["reveal_start"] = time.time()
        ctx["sd_phase"] = "hold_colored"
        ctx["sd_phase_start"] = ctx["reveal_start"]
        # Clear caches when starting fresh
        ctx.pop("_bordered_grid", None)
        ctx.pop("colored_grid", None)
        ctx.pop("filtered_grid", None)
        ctx.pop("subtle_grid", None)

        grid = _make_colored_grid(ctx)
        return _prepare_reveal_grid(grid, ctx)

    def _grid_for_phase(self, ctx, phase, elapsed):
        """Build the grid for the current phase/elapsed-in-phase."""
        if phase == "hold_colored":
            return _make_colored_grid(ctx)
        if phase == "hold_filtered":
            return _make_filtered_grid(ctx)
        if phase == "to_filtered":
            alpha = min(elapsed / self.TRANSITION, 1.0)
            return _make_dissolve_grid(ctx, alpha)
        if phase == "to_colored":
            alpha = min(elapsed / self.TRANSITION, 1.0)
            return _make_dissolve_grid(ctx, 1.0 - alpha)
        return None

    def update(self, ctx, just_pressed):
        now = time.time()

        if now - ctx.get("reveal_start", now) >= self.TOTAL or just_pressed:
            return None, True

        phase = ctx.get("sd_phase", "hold_colored")
        elapsed = now - ctx.get("sd_phase_start", now)
        duration = self.HOLD if phase.startswith("hold") else self.TRANSITION

        if elapsed >= duration:
            next_phase = {
                "hold_colored": "to_filtered",
                "to_filtered": "hold_filtered",
                "hold_filtered": "to_colored",
                "to_colored": "hold_colored",
            }[phase]
            ctx["sd_phase"] = next_phase
            ctx["sd_phase_start"] = now
            grid = self._grid_for_phase(ctx, next_phase, 0.0)
        elif phase in ("to_filtered", "to_colored"):
            grid = self._grid_for_phase(ctx, phase, elapsed)
        else:
            return None, False

        return _prepare_reveal_grid(grid, ctx), False


class StandardReveal(RevealStrategy):
    """Colored overlay 5s → filtered images 30s"""

    def get_name(self) -> str:
        return "Standard"

    def get_description(self) -> str:
        return "Colored overlay 5s → filtered images 30s"

    def get_initial_grid(self, ctx):
        ctx["reveal_stage"] = "colored"
        ctx["reveal_start"] = time.time()
        # Clear caches when starting fresh
        ctx.pop("_bordered_grid", None)
        ctx.pop("colored_grid", None)
        ctx.pop("filtered_grid", None)
        ctx.pop("subtle_grid", None)

        grid = _make_colored_grid(ctx)
        return _prepare_reveal_grid(grid, ctx)

    def update(self, ctx, just_pressed):
        now = time.time()

        if ctx.get("reveal_stage") == "colored":
            if now - ctx.get("reveal_start", now) >= 5.0:
                ctx["reveal_stage"] = "filtered"
                ctx["reveal_start"] = now
                grid = _make_filtered_grid(ctx)
                return _prepare_reveal_grid(grid, ctx), False
            return None, False

        if ctx.get("reveal_stage") == "filtered":
            if now - ctx.get("reveal_start", now) >= cfg.REVEAL_DURATION or just_pressed:
                return None, True

            grid = _make_filtered_grid(ctx)
            return _prepare_reveal_grid(grid, ctx), False

        return None, False


class SubtleReveal(RevealStrategy):
    """All 4 colored images with low opacity overlays"""

    def get_name(self) -> str:
        return "Subtle"

    def get_description(self) -> str:
        return "Colored overlays at 50% opacity"

    def get_initial_grid(self, ctx):
        ctx["reveal_start"] = time.time()
        # Clear caches when starting fresh
        ctx.pop("_bordered_grid", None)
        ctx.pop("colored_grid", None)
        ctx.pop("filtered_grid", None)
        ctx.pop("subtle_grid", None)

        grid = _make_subtle_grid(ctx)
        return _prepare_reveal_grid(grid, ctx)

    def update(self, ctx, just_pressed):
        now = time.time()

        if now - ctx.get("reveal_start", now) >= cfg.REVEAL_DURATION or just_pressed:
            return None, True

        grid = _make_subtle_grid(ctx)
        return _prepare_reveal_grid(grid, ctx), False


class DissolveReveal(RevealStrategy):
    """Crossfades from colored diff to filtered result over 3s"""

    DURATION = 3.0

    def get_name(self) -> str:
        return "Dissolve"

    def get_description(self) -> str:
        return "Smooth crossfade from colored diff to filtered result"

    def get_initial_grid(self, ctx):
        ctx["reveal_start"] = time.time()
        ctx["dissolve_done"] = False
        # Clear caches when starting fresh
        ctx.pop("_bordered_grid", None)
        ctx.pop("colored_grid", None)
        ctx.pop("filtered_grid", None)
        ctx.pop("subtle_grid", None)

        grid = _make_dissolve_grid(ctx, 0.0)
        return _prepare_reveal_grid(grid, ctx)

    def update(self, ctx, just_pressed):
        now = time.time()
        elapsed = now - ctx.get("reveal_start", now)

        if elapsed < self.DURATION:
            alpha = elapsed / self.DURATION
            grid = _make_dissolve_grid(ctx, alpha)
            return _prepare_reveal_grid(grid, ctx), False

        if not ctx.get("dissolve_done"):
            ctx["dissolve_done"] = True
            ctx["hold_start"] = now
            grid = _make_filtered_grid(ctx)
            return _prepare_reveal_grid(grid, ctx), False

        if now - ctx.get("hold_start", now) >= cfg.REVEAL_DURATION or just_pressed:
            return None, True

        grid = _make_filtered_grid(ctx)
        return _prepare_reveal_grid(grid, ctx), False


REVEAL_STRATEGIES = {
    "standard": StandardReveal(),
    "dissolve": DissolveReveal(),
    "subtle": SubtleReveal(),
    "slideshow": SlideshowReveal(),
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