import os
import subprocess
import random
import cv2
import numpy as np
import math
from typing import List, Tuple
from PIL import ImageFont, ImageDraw, Image

_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_font_cache: dict = {}


def random_list(items):
    rng = np.random.default_rng()
    indices = rng.permutation(len(items))
    return [items[i] for i in indices]

def random_filter(items):
    k = min(3, len(items))
    return random.sample(items, k)


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _font_cache:
        _font_cache[size] = ImageFont.truetype(_FONT_PATH, size)
    return _font_cache[size]


def print_text(frame, text,
               font_scale=2.5, thickness=3, color=(255, 255, 255),
               position="top", style="bar"):
    """
    Renders Unicode text (including Umlauts) via Pillow so ä/ö/ü display correctly.
    style options:    "pill", "outline", "bar"
    position options: "top", "center", "bottom"
    color: BGR tuple (same convention as the rest of the codebase)
    font_scale=2.5 → ~125px → ~7.2cm on your 44 PPI screen (good title size)
    """
    font_size = max(12, int(font_scale * 50))
    font = _get_font(font_size)

    fh, fw = frame.shape[:2]
    pad_x, pad_y = 18, 10
    margin = 24

    # Measure text
    dummy_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    _, descent = font.getmetrics()

    x = (fw - text_w) // 2

    if position == "top":
        y = margin
    elif position == "bottom":
        y = fh - margin - text_h - descent
    else:
        y = (fh - text_h) // 2

    # BGR -> RGB for Pillow
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img, "RGBA")

    # Pillow wants RGB, codebase passes BGR
    r, g, b = color[2], color[1], color[0]

    if style == "pill":
        rx, ry = x - pad_x, y - pad_y
        rw = text_w + pad_x * 2
        rh = text_h + pad_y * 2 + descent
        draw.rectangle([rx, ry, rx + rw, ry + rh], fill=(30, 30, 30, 165))
        draw.text((x, y), text, font=font, fill=(r, g, b, 255))

    elif style == "outline":
        for dx, dy in [(-2,-2),(2,-2),(-2,2),(2,2),(0,-2),(0,2),(-2,0),(2,0)]:
            draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0, 255))
        draw.text((x, y), text, font=font, fill=(r, g, b, 255))

    elif style == "bar":
        bar_y = y - pad_y
        bar_h = text_h + pad_y * 2 + descent
        draw.rectangle([0, bar_y, fw, bar_y + bar_h], fill=(0, 0, 0, 140))
        draw.text((x, y), text, font=font, fill=(r, g, b, 255))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def get_screen_resolution():
    try:
        env = {**os.environ, 'DISPLAY': ':0'}
        output = subprocess.check_output(
            "xrandr | grep '*' | awk '{print $1}'",
            shell=True, env=env
        ).decode().strip().split('\n')[0]
        w, h = output.split('x')
        return int(w), int(h)
    except:
        return 1920, 1200


def _fit_image(img, target_w, target_h):
    """Scale-to-fill + centre-crop so the image exactly fills the cell."""
    ih, iw = img.shape[:2]
    scale = max(target_w / iw, target_h / ih)
    new_w = int(round(iw * scale))
    new_h = int(round(ih * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Centre crop
    x0 = (new_w - target_w) // 2
    y0 = (new_h - target_h) // 2
    return resized[y0:y0 + target_h, x0:x0 + target_w]

"""
Drawing utilities for wrinkle rendering.
"""


def calculate_wrinkle_color(skin_colors: List[Tuple[int, int, int]], darker: bool = False) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Calculate wrinkle color with highlight+shadow pairing."""
    if not skin_colors:
        return (100, 80, 60), (120, 100, 80)

    try:
        avg_b = int(np.mean([c[0] for c in skin_colors]))
        avg_g = int(np.mean([c[1] for c in skin_colors]))
        avg_r = int(np.mean([c[2] for c in skin_colors]))

        avg_b = max(0, min(255, avg_b))
        avg_g = max(0, min(255, avg_g))
        avg_r = max(0, min(255, avg_r))

        shadow_factor = 0.65 if darker else 0.85
        highlight_factor = 1.10 if darker else 1.05

        shadow_color = (int(avg_b * shadow_factor), int(avg_g * shadow_factor), int(avg_r * shadow_factor))
        highlight_color = (int(min(255, avg_b * highlight_factor)),
                          int(min(255, avg_g * highlight_factor)),
                          int(min(255, avg_r * highlight_factor)))

        return shadow_color, highlight_color
    except:
        return (100, 80, 60), (120, 100, 80)


def cubic_bezier(p0: Tuple[float, float], p1: Tuple[float, float],
                 p2: Tuple[float, float], p3: Tuple[float, float], t: float) -> Tuple[float, float]:
    """Calculate point on cubic Bezier curve."""
    mt = 1 - t
    x = mt ** 3 * p0[0] + 3 * mt ** 2 * t * p1[0] + 3 * mt * t ** 2 * p2[0] + t ** 3 * p3[0]
    y = mt ** 3 * p0[1] + 3 * mt ** 2 * t * p1[1] + 3 * mt * t ** 2 * p2[1] + t ** 3 * p3[1]
    return (x, y)


def generate_bezier_points(points: List[Tuple[int, int]], num_points: int = 30) -> List[Tuple[int, int]]:
    """Generate smooth Bezier curve through given points."""
    if len(points) < 2:
        return points
    if len(points) == 2:
        return points

    bezier_points = []
    for i in range(len(points) - 1):
        p0 = points[max(0, i - 1)]
        p1 = points[i]
        p2 = points[i + 1]
        p3 = points[min(len(points) - 1, i + 2)]
        cp1 = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

        for t in np.linspace(0, 1, num_points):
            point = cubic_bezier(p0, p1, cp1, p2, t)
            bezier_points.append((int(point[0]), int(point[1])))

    unique_points = []
    for p in bezier_points:
        if not unique_points or unique_points[-1] != p:
            unique_points.append(p)
    return unique_points


def add_broken_segments(points: List[Tuple[int, int]], broken_chance: float = 0.15) -> List[List[Tuple[int, int]]]:
    """Split line into broken segments by randomly removing sections."""
    if len(points) < 10 or broken_chance <= 0:
        return [points]

    segments = []
    current_segment = []

    for i, point in enumerate(points):
        if i > 0 and np.random.random() < broken_chance:
            if len(current_segment) > 1:
                segments.append(current_segment)
            current_segment = [point]
        else:
            current_segment.append(point)

    if len(current_segment) > 1:
        segments.append(current_segment)
    return segments if segments else [points]


def draw_tapered_line(image: np.ndarray, points: List[Tuple[int, int]],
                      shadow_color: Tuple[int, int, int],
                      highlight_color: Tuple[int, int, int],
                      max_width: int = 3):
    """Draw tapered line with shadow and highlight pairing."""
    if len(points) < 2:
        return

    total_points = len(points)
    for i in range(total_points - 1):
        t = i / (total_points - 1) if total_points > 1 else 0.5
        width_factor = math.sin(t * math.pi)
        current_width = max(1, int(max_width * width_factor))

        p1, p2 = points[i], points[i + 1]
        for w in range(current_width):
            cv2.line(image, p1, p2, shadow_color, 1, cv2.LINE_AA)

        highlight_width = max(1, current_width // 2)
        for w in range(highlight_width):
            cv2.line(image, p1, p2, highlight_color, 1, cv2.LINE_AA)