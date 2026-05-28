import random
import time
import cv2
import numpy as np
import config as cfg
from landmarks import REGIONS, LEFT_EYE_INNER, RIGHT_EYE_INNER, LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR, LEFT_EYEBROW, \
    RIGHT_EYEBROW, NOSTRIL_AREA, LIPS_OUTER
import logging
from utils import random_filter
from webserver import _disabled_filters

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_WARP_MAP_CACHE_MAX = 4


def get_filters(enhancer, wrinkles):
    all_filters = [
        ("Micro_proportion_shifts", enhancer.apply_proportion_shifts),
        ("Skin_color_adjustment", enhancer.apply_skin_tweaks_new),
        ("Geometric_micro", enhancer.apply_geometric_edits),
        ("Whiten_sclera", enhancer.whiten_sclera),
        ("Widen_face",
         lambda img, lm, **kw: enhancer.widen_or_slim_face(img, lm, strength=cfg.FACE_WIDEN_STRENGTH, slim=False)),
        ("Slim_face",
         lambda img, lm, **kw: enhancer.widen_or_slim_face(img, lm, strength=cfg.FACE_SLIM_STRENGTH, slim=True)),
        ("Random_mole", enhancer.add_random_moles),
        ("Accentuate_cheekbone", enhancer.contour_cheekbones),
        ("Remove_eye_bags", enhancer.remove_eye_bags),
        ("Freckles", enhancer.add_freckles),
        ("Symmetrical_face", enhancer.apply_symmetry),
        ("Draw_wrinkles", wrinkles.draw_all_wrinkles)
    ]

    return [f for f in all_filters if f[0] not in _disabled_filters]


def prepare_frame(frame):
    frame = cv2.flip(frame, 1)

    frame, was_adjusted = adjust_dark_image(frame)
    if was_adjusted:
        logger.info("Dark image detected, brightness adjusted")

    return frame


def _unpack_filter_result(name, result):
    """Unpack filter result into (image, mole_positions, freckle_positions)."""
    if name == "Random_mole" and isinstance(result, tuple):
        return result[0], result[1], None
    if name == "Freckles" and isinstance(result, tuple):
        return result[0], None, result[1]
    return result, None, None


def _make_diff_or_wrinkle(wrinkles, frame, img, landmarks, name, mole_positions, freckle_pos):
    """Handle diff overlay, with wrinkle filter as a special case."""
    if name == "Draw_wrinkles":
        wrinkles.debug = True
        diff_img = wrinkles.draw_all_wrinkles(img, landmarks)
        wrinkles.debug = False
        return diff_img, True
    return make_diff_overlay(frame, img, landmarks,
                             mole_points=mole_positions,
                             freckles=freckle_pos)


def _apply_filters(enhancer, wrinkles, frame, landmarks, progress_callback=None, **kwargs):
    filters = get_filters(enhancer, wrinkles)
    chosen = random_filter(filters)
    remaining = [f for f in filters if f not in chosen]

    t_total = time.time()
    n_filters = len(chosen)

    used_keys = ["Original"]
    for filter_index, (name, function) in enumerate(chosen):
        logger.info(f"  Processing: {name}...")

        if progress_callback:
            progress_callback(filter_index, n_filters)

        t0 = time.time()
        result = function(frame, landmarks, **kwargs)
        elapsed = time.time() - t0
        logger.info(f"  ⏱  {name} took {elapsed:.3f}s")

        img, mole_positions, freckle_pos = _unpack_filter_result(name, result)
        diff_img, any_change = _make_diff_or_wrinkle(wrinkles, frame, img, landmarks, name, mole_positions, freckle_pos)

        if not any_change:
            logger.info(f"  '{name}' had no effect, trying a replacement...")
            replaced = False
            for alt_name, alt_function in remaining:
                logger.info(f"  Processing replacement: {alt_name}...")

                t0 = time.time()
                alt_result = alt_function(frame, landmarks, **kwargs)
                elapsed = time.time() - t0
                logger.info(f"  ⏱  {alt_name} took {elapsed:.3f}s")

                alt_img, mole_positions, freckle_pos = _unpack_filter_result(alt_name, alt_result)
                alt_diff, alt_any_change = _make_diff_or_wrinkle(wrinkles, frame, alt_img, landmarks, alt_name, mole_positions, freckle_pos)

                remaining = [f for f in remaining if f[0] != alt_name]
                if alt_any_change:
                    name, img, diff_img = alt_name, alt_img, alt_diff
                    replaced = True
                    break

            if not replaced:
                logger.info(f"  No replacement found for '{name}', skipping.")
                continue

        cv2.imwrite(cfg.IMAGE_PATHS[name], img)
        cv2.imwrite(cfg.DIFF_PATHS[name], diff_img)
        used_keys.append(name)

    logger.info(f"  ⏱  Total filter time: {time.time() - t_total:.3f}s")
    return used_keys


def adjust_dark_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    current = np.mean(lab[:, :, 0])
    logger.debug(f"    Image luminance: {current:.1f}")
    if current < cfg.DARK_IMAGE_THRESHOLD:
        l = lab[:, :, 0].astype(np.float32)
        boost = cfg.TARGET_LUMINANCE - current
        l = l + boost * 0.7
        gamma = max(cfg.GAMMA_MIN, 1.0 - (cfg.DARK_IMAGE_THRESHOLD - current) / 150)
        l = np.power(np.clip(l / 255.0, 0, 1), gamma) * 255.0
        l = np.clip(l, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=cfg.CLAHE_CLIP_LIMIT, tileGridSize=cfg.CLAHE_TILE_SIZE)
        l2 = clahe.apply(l)
        lab[:, :, 0] = cv2.addWeighted(l, 1.0 - cfg.CLAHE_BLEND_WEIGHT, l2, cfg.CLAHE_BLEND_WEIGHT, 0)
        adj = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        logger.debug(f"    Adjusted luminance: {np.mean(cv2.cvtColor(adj, cv2.COLOR_BGR2LAB)[:, :, 0]):.1f}")
        return adj, True
    return image, False


def _draw_point_markers(result, landmarks, points, color=(255, 255, 0)):
    """Draw circles at given points, sized relative to eye distance."""
    if not points:
        return result
    left_eye = np.array(landmarks[LEFT_EYE_INNER])
    right_eye = np.array(landmarks[RIGHT_EYE_INNER])
    radius = max(5, int(np.linalg.norm(left_eye - right_eye) * 0.08))
    for (mx, my) in points:
        cv2.circle(result, (int(mx), int(my)), radius, color, thickness=3, lineType=cv2.LINE_AA)
    return result


def make_diff_overlay(original, enhanced, landmarks, amplify=30, threshold=2, mole_points=None, freckles=None):
    diff = cv2.absdiff(original, enhanced).astype(np.float32)
    gray_diff = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    result = enhanced.copy()
    h, w = result.shape[:2]

    any_change = False

    for region_name, region_data in REGIONS.items():
        indices = region_data["indices"]
        color = region_data["color"]
        pts = np.array([(int(landmarks[i][0]), int(landmarks[i][1]))
                        for i in indices if i < len(landmarks)], dtype=np.int32)
        if len(pts) < 3:
            continue

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        mean_diff = cv2.mean(gray_diff, mask=mask)[0]

        if mean_diff < threshold:
            continue

        any_change = True
        opacity = min(mean_diff * amplify / 255.0, 0.85)

        if "Lips" in region_name:
            opacity = min(opacity * 2, 0.9)

        overlay = result.copy()
        cv2.fillPoly(overlay, [pts], color)
        result = cv2.addWeighted(overlay, opacity, result, 1 - opacity, 0)
        cv2.polylines(result, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

    if mole_points:
        any_change = True
        result = _draw_point_markers(result, landmarks, mole_points)

    if freckles:
        any_change = True
        result = _draw_point_markers(result, landmarks, freckles)

    return result, any_change


def _get_points(lm, idx):
    return np.array([lm[i] for i in idx if i < len(lm)], dtype=np.int32)


def _center_of(lm, idx):
    pts = np.array([lm[i] for i in idx if i < len(lm)], dtype=np.float64)
    return pts.mean(axis=0)


def _make_polygon_mask(shape, points, feather=0):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    if feather > 0:
        k = feather * 2 + 1
        mask = cv2.GaussianBlur(mask.astype(np.float32), (k, k), 0)
        mask = mask / max(mask.max(), 1.0)
    else:
        mask = mask.astype(np.float32) / 255.0
    return mask


def _make_ellipse_mask(shape, center, axes, feather=15):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    k = feather * 2 + 1
    mask = cv2.GaussianBlur(mask.astype(np.float32), (k, k), 0)
    return mask / max(mask.max(), 1.0)


# ---------------------------------------------------------------------------
# Warp-map cache — np.meshgrid for a given (h, w) is expensive to rebuild on
# every call.  Cache the base X/Y maps keyed by image size; callers modify
# copies so the originals stay clean.
# ---------------------------------------------------------------------------
_warp_map_cache: dict = {}  # (h, w) -> (mapX_base, mapY_base)


def _get_base_maps(h: int, w: int):
    """Return cached (mapX, mapY) float32 base grids for the given size."""
    key = (h, w)
    if key not in _warp_map_cache:
        if len(_warp_map_cache) >= _WARP_MAP_CACHE_MAX:
            _warp_map_cache.pop(next(iter(_warp_map_cache)))
        mx, my = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))
        _warp_map_cache[key] = (mx, my)
    return _warp_map_cache[key]


def _local_translate_warp(image, center, shift, radius):
    h, w = image.shape[:2]
    mapX_base, mapY_base = _get_base_maps(h, w)
    mapX = mapX_base.copy()
    mapY = mapY_base.copy()
    cx, cy = float(center[0]), float(center[1])
    dx, dy = float(shift[0]), float(shift[1])
    dX, dY = mapX - cx, mapY - cy
    dist = np.sqrt(dX ** 2 + dY ** 2)
    mask = np.clip(1.0 - dist / radius, 0, 1) ** 2
    return cv2.remap(image, (mapX - dx * mask).astype(np.float32),
                     (mapY - dy * mask).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _local_scale_warp(image, center, scale, radius):
    h, w = image.shape[:2]
    mapX_base, mapY_base = _get_base_maps(h, w)
    mapX = mapX_base.copy()
    mapY = mapY_base.copy()
    cx, cy = float(center[0]), float(center[1])
    dX, dY = mapX - cx, mapY - cy
    dist = np.sqrt(dX ** 2 + dY ** 2)
    mask = np.clip(1.0 - dist / radius, 0, 1) ** 2
    s = 1.0 + (1.0 / scale - 1.0) * mask
    return cv2.remap(image, (cx + dX * s).astype(np.float32),
                     (cy + dY * s).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def get_skin_color(image, landmarks, zone_indices):
    samples = []
    h, w = image.shape[:2]
    for idx in zone_indices:
        if idx < len(landmarks):
            x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
            if 2 <= x < w - 2 and 2 <= y < h - 2:
                patch = image[y - 2:y + 3, x - 2:x + 3]
                samples.append(patch.reshape(-1, 3).mean(axis=0))
    if not samples:
        return np.array([80, 60, 50], dtype=np.float32)
    return np.mean(samples, axis=0).astype(np.float32)


def pick_mole_position(landmarks, zone_indices):
    """Pick a random position inside a landmark zone with jitter."""
    pts = [landmarks[i] for i in zone_indices if i < len(landmarks)]
    if len(pts) < 2:
        return None
    chosen = random.sample(pts, min(3, len(pts)))
    cx = int(np.mean([p[0] for p in chosen]))
    cy = int(np.mean([p[1] for p in chosen]))
    jitter = random.randint(3, 12)
    cx += random.randint(-jitter, jitter)
    cy += random.randint(-jitter, jitter)
    return (cx, cy)


def draw_mole(image, center, skin_color, eye_dist):
    """Draw a single realistic mole with natural color variation."""
    h, w = image.shape[:2]
    cx, cy = center

    if not (5 <= cx < w - 5 and 5 <= cy < h - 5):
        return image

    result = image.copy()

    base_radius = max(1, int(eye_dist * random.uniform(0.005, 0.01)))

    darkness = random.uniform(0.35, 0.55)
    mole_color = skin_color * darkness
    mole_color[2] = min(255, mole_color[2] + random.uniform(5, 20))
    mole_color = np.clip(mole_color, 0, 255)

    mask_size = base_radius * 4 + 1
    mask = np.zeros((mask_size * 2 + 1, mask_size * 2 + 1), dtype=np.float32)
    cv2.circle(mask, (mask_size, mask_size), base_radius, 1.0, -1)

    blur_k = base_radius * 2 + 1
    if blur_k % 2 == 0:
        blur_k += 1
    mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)
    mask = mask / max(mask.max(), 1e-5)

    if random.random() > 0.4:
        noise = np.random.normal(1.0, 0.15, mask.shape).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (blur_k, blur_k), 0)
        mask = mask * np.clip(noise, 0.7, 1.3)
        mask = mask / max(mask.max(), 1e-5)

    opacity = random.uniform(0.6, 0.9)
    mask = mask * opacity

    y1 = cy - mask_size
    y2 = cy + mask_size + 1
    x1 = cx - mask_size
    x2 = cx + mask_size + 1

    my1 = max(0, -y1)
    mx1 = max(0, -x1)
    y1 = max(0, y1)
    x1 = max(0, x1)
    y2 = min(h, y2)
    x2 = min(w, x2)
    my2 = my1 + (y2 - y1)
    mx2 = mx1 + (x2 - x1)

    if y2 <= y1 or x2 <= x1:
        return image

    roi_mask = mask[my1:my2, mx1:mx2]
    roi_mask_3ch = roi_mask[:, :, np.newaxis]

    roi = result[y1:y2, x1:x2].astype(np.float32)
    mole_layer = np.full_like(roi, mole_color)

    if base_radius >= 3 and random.random() > 0.3:
        center_mask = np.zeros_like(roi_mask)
        cr = max(1, base_radius // 2)
        local_cx = cx - x1
        local_cy = cy - y1
        if 0 <= local_cx < roi_mask.shape[1] and 0 <= local_cy < roi_mask.shape[0]:
            cv2.circle(center_mask, (local_cx, local_cy), cr, 1.0, -1)
            center_mask = cv2.GaussianBlur(center_mask, (cr * 2 + 1, cr * 2 + 1), 0)
            center_mask = center_mask / max(center_mask.max(), 1e-5) * 0.3
            darker_center = mole_color * 0.7
            mole_layer = mole_layer * (1 - center_mask[:, :, np.newaxis]) + \
                         darker_center * center_mask[:, :, np.newaxis]

    blended = roi * (1 - roi_mask_3ch) + mole_layer * roi_mask_3ch
    result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

    return result


def build_exclusion_mask(image, landmarks, hair_mask=None, extra_landmark_sets=None):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    standard_regions = [
        LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR,
        LEFT_EYEBROW, RIGHT_EYEBROW,
        LIPS_OUTER,
        NOSTRIL_AREA,
    ]

    all_regions = standard_regions[:]
    if extra_landmark_sets:
        if isinstance(extra_landmark_sets, list):
            all_regions.extend(extra_landmark_sets)
        else:
            all_regions.append(extra_landmark_sets)

    for idx_list in all_regions:
        pts = [(int(landmarks[i][0]), int(landmarks[i][1])) for i in idx_list if i < len(landmarks)]
        if len(pts) >= 3:
            cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)

    if hair_mask is not None:
        if hair_mask.shape[:2] != (h, w):
            hair_mask = cv2.resize(hair_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        hair_binary = (hair_mask > 0.5).astype(np.uint8) * 255
        hair_margin = 5
        hair_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hair_margin * 2 + 1, hair_margin * 2 + 1))
        hair_dilated = cv2.dilate(hair_binary, hair_kernel)
        mask = cv2.bitwise_or(mask, hair_dilated)

    return mask


def _warp_triangle(src, dst, tri_src, tri_dst):
    r1 = cv2.boundingRect(np.float32([tri_src]))
    r2 = cv2.boundingRect(np.float32([tri_dst]))
    t1 = [(p[0] - r1[0], p[1] - r1[1]) for p in tri_src]
    t2 = [(p[0] - r2[0], p[1] - r2[1]) for p in tri_dst]
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2), (1, 1, 1), 16, 0)
    img_crop = src[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    if img_crop.size == 0:
        return
    M = cv2.getAffineTransform(np.float32(t1), np.float32(t2))
    warped = cv2.warpAffine(img_crop, M, (r2[2], r2[3]),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)
    dst_crop = dst[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]
    dst_crop[:] = dst_crop * (1 - mask) + warped * mask