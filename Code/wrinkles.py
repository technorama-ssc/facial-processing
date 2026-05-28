import cv2
import numpy as np
import math
from landmarks import REGIONS
import config
from hair_detection import Hair
from hessian_detector import HessianWrinkleDetector
from utils import (calculate_wrinkle_color, generate_bezier_points,
                   add_broken_segments, draw_tapered_line)
from landmarks import WRINKLES

def point_in_region(pt, region_pts):
    return cv2.pointPolygonTest(region_pts, pt, False) >= 0


def get_wrinkle_color_for_point(pt, landmarks):
    for region_name, region_data in REGIONS.items():
        indices = region_data["indices"]
        pts = np.array([landmarks[i] for i in indices if i < len(landmarks)], dtype=np.int32)
        if len(pts) >= 3:
            if point_in_region(pt, pts):
                return region_data["color"]
    return (60, 80, 100)


class CombinedWrinkleDrawer:
    def __init__(self):
        self.hair_segmenter = Hair()

        self.face_angle = 0
        self.hessian_detector = HessianWrinkleDetector(
            scale_range=(config.HESSIAN_SCALE_MIN, config.HESSIAN_SCALE_MAX),
            scale_step=config.HESSIAN_SCALE_STEP,
            sensitivity=config.HESSIAN_SENSITIVITY
        )

        self.TARGET_BRIGHTNESS = 120
        self.MIN_BRIGHTNESS = 80

        self.debug = False

    def is_valid_point(self, pt, im):
        h, w = im.shape[:2]
        return (pt is not None and
                len(pt) >= 2 and
                0 < pt[0] < w and
                0 < pt[1] < h)

    def check_forehead_has_hair(self, image, landmarks):
        if not self.hair_segmenter:
            return False

        left, top, right, bottom = self.get_forehead_bounds(landmarks)
        if left == 0 and right == 0:
            return False

        raw = self.hair_segmenter.get_hair_mask_sync(image)
        if raw is None:
            if config.VERBOSE:
                print("  ℹ No hair mask — assuming clear")
            return False

        h, w = image.shape[:2]
        fw = right - left
        m = int(fw * 0.30)
        sx = int(max(0, left + m))
        ex = int(min(w, right - m))

        sy = int(max(0, top))
        ey = int(min(h, bottom))

        if sx >= ex or sy >= ey:
            return False

        reg = raw[sy:ey, sx:ex]
        if reg.size == 0:
            return False

        pct = np.sum(reg > 0) / reg.size * 100
        has = pct > 1.0

        if config.VERBOSE:
            print(f"  {'⚠' if has else '✓'} Forehead hair: {pct:.1f}%" +
                  (" — BANGS" if has else " — clear"))

        return has

    def auto_adjust_brightness(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        avg = np.mean(l)
        if avg < self.MIN_BRIGHTNESS:
            adj = self.TARGET_BRIGHTNESS - avg
            factor = 1.5 if avg < 50 else (1.3 if avg < 70 else 1.15)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            leq = clahe.apply(l)
            blend = min(0.7, adj / 100)
            lad = cv2.addWeighted(l, 1 - blend, leq, blend, 0)
            lad = np.clip(lad.astype(np.float32) + adj * factor, 0, 255).astype(np.uint8)
            return cv2.cvtColor(cv2.merge([lad, a, b]), cv2.COLOR_LAB2BGR)
        return image

    def detect_face_angle(self, lm):
        lo, li, ri, ro = 33, 133, 362, 263
        if lo >= len(lm) or ro >= len(lm): return 0
        lx = (lm[lo][0] + lm[li][0]) / 2
        ly = (lm[lo][1] + lm[li][1]) / 2
        rx = (lm[ri][0] + lm[ro][0]) / 2
        ry = (lm[ri][1] + lm[ro][1]) / 2
        return math.degrees(math.atan2(ry - ly, rx - lx))

    def rotate_point(self, pt, center, angle):
        rad = math.radians(angle)
        x, y = pt
        cx, cy = center
        tx, ty = x - cx, y - cy
        rx = tx * math.cos(rad) - ty * math.sin(rad)
        ry = tx * math.sin(rad) + ty * math.cos(rad)
        return (int(rx + cx), int(ry + cy))

    def get_forehead_bounds(self, lm):
        lb = [70, 63, 105, 66, 107]
        rb = [336, 296, 334, 293, 276]
        tf = [10, 151, 9, 338, 297]
        bys = [lm[i][1] for i in lb + rb if i < len(lm)]
        tys = [lm[i][1] for i in tf if i < len(lm)]
        if not bys or not tys: return (0, 0, 0, 0)
        xs = [lm[i][0] for i in tf + lb + rb if i < len(lm)]
        return (
            int(min(xs)),
            int(min(tys)),
            int(max(xs)),
            int(max(bys))
        )

    def get_face_center(self, lm):
        return lm[1] if len(lm) > 1 else (0, 0)

    def get_area_skin_color(self, im, cx, cy, r=20):
        h, w = im.shape[:2]
        cols = []
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h:
                    px = im[ny, nx]
                    if len(px) == 3: cols.append(tuple(px))
        return cols

    def build_eyebrow_mask(self, im, lm):
        pad = 6
        L = [55, 65, 52, 53, 46, 70, 63, 105, 66, 107]
        R = [285, 336, 296, 334, 293, 300, 276, 283, 282, 295]
        h, w = im.shape[:2]
        mask = np.full((h, w), 255, dtype=np.uint8)
        for idxs in (L, R):
            pts = [lm[i] for i in idxs if i < len(lm)]
            if len(pts) < 3:
                continue
            pts_array = np.array(pts, dtype=np.int32).reshape(-1, 2)
            hull = cv2.convexHull(pts_array)
            cv2.fillPoly(mask, [hull], 0)
            mask = cv2.erode(mask, np.ones((pad * 2 + 1, pad * 2 + 1), np.uint8), iterations=1)
        return mask

    def generate_forehead_wrinkle_points(self, lm, pos, curve):
        l, t, r, b = self.get_forehead_bounds(lm)
        if l == 0 and r == 0: return []
        yb = t + (b - t) * pos
        var = 8 if pos > 0.6 else (6 if pos > 0.35 else 4)
        fw = r - l
        m = fw * 0.30
        sx, ex = l + m, r - m
        ww = ex - sx
        pts = []
        fc = self.get_face_center(lm)
        for i in range(41):
            tt = i / 40
            x = sx + tt * ww
            dip = -abs(curve) * (1 - (2 * tt - 1) ** 2) * (ww * 0.12)
            wave = math.sin(tt * math.pi * 3) * var * (1 - abs(tt - 0.5) * 1.5)
            asym = math.sin(tt * math.pi * 1.5) * 3
            y = yb + dip + wave + asym
            y = max(t + 5, min(b - 5, y))
            x = max(l + 5, min(r - 5, x))
            pts.append(self.rotate_point((int(x), int(y)), fc, self.face_angle))
        return pts

    def _draw_colored_wrinkle(self, im, pts, color, broken=0.15, max_width=3):
        """Debug mode: draw wrinkle using explicit region color."""
        if len(pts) < 2:
            return
        smooth = generate_bezier_points(pts, 25)
        segs = add_broken_segments(smooth, broken)
        for segment in segs:
            if len(segment) < 2:
                continue
            total = len(segment)
            for i in range(total - 1):
                t = i / (total - 1) if total > 1 else 0.5
                width_factor = math.sin(t * math.pi)
                current_width = max(1, int(max_width * width_factor))
                p1, p2 = segment[i], segment[i + 1]
                shadow = tuple(max(0, int(c * 0.6)) for c in color)
                highlight = tuple(min(255, int(c * 1.3)) for c in color)
                for _ in range(current_width):
                    cv2.line(im, p1, p2, shadow, 1, cv2.LINE_AA)
                hl_width = max(1, current_width // 2)
                for _ in range(hl_width):
                    cv2.line(im, p1, p2, highlight, 1, cv2.LINE_AA)

    def draw_wrinkle(self, im, pts, darker=False, broken=0.15, max_width=3,
                     extra_prominent=False, is_forehead=False, landmarks=None):
        if len(pts) < 2:
            return

        w = max_width + 2 if is_forehead else (max_width + 4 if extra_prominent else max_width + 3)

        # ── DEBUG MODE: colored wrinkles by facial region ──────────────────────
        if self.debug and landmarks is not None:
            mid_pt = pts[len(pts) // 2]
            color = get_wrinkle_color_for_point(mid_pt, landmarks)
            self._draw_colored_wrinkle(im, pts, color, broken, w)
            return

        # ── NORMAL MODE: skin-tone derived shadow/highlight ────────────────────
        smooth = generate_bezier_points(pts, 25)
        segs = add_broken_segments(smooth, broken)
        h, img_w = im.shape[:2]

        # These two lines are missing — add them back:
        mid = len(smooth) // 2
        skin = self.get_area_skin_color(im, smooth[mid][0], smooth[mid][1], 15) if mid < len(smooth) else []
        sh, hl = calculate_wrinkle_color(skin, darker)
        if extra_prominent:
            sh = (int(sh[0] * .85), int(sh[1] * .85), int(sh[2] * .85))

        for s in segs:
            clipped = [p for p in s if 0 <= p[0] < img_w and 0 <= p[1] < h]
            if len(clipped) >= 2:
                draw_tapered_line(im, clipped, sh, hl, w)

    def draw_hessian_wrinkles(self, im, lm, debug_filename=None):
        canvas = im.copy()
        det, has = self.hessian_detector.detect_wrinkles(im, lm)
        skip_fh = self.check_forehead_has_hair(im, lm) if has and any(r == 'forehead' for r in det) else False
        if has:
            for reg, paths in det.items():
                if reg not in config.HESSIAN_REGION_STYLES: continue
                st = config.HESSIAN_REGION_STYLES[reg]
                extra = reg in ['under_eye_left', 'under_eye_right', 'nasolabial_left', 'nasolabial_right',
                                'marionette_left', 'marionette_right']
                is_fh = (reg == 'forehead')
                if is_fh and skip_fh: continue
                for p in paths:
                    filtered = [pt for pt in p if self.is_valid_point(pt, canvas)]
                    if len(filtered) >= 2:
                        self.draw_wrinkle(canvas, filtered, st['darker'], st['broken_chance'],
                                          st['width'], extra, is_fh, landmarks=lm)
        return canvas, has

    def draw_manual_wrinkles(self, im, lm, debug_filename=None):
        canvas = im.copy()
        skip_fh = self.check_forehead_has_hair(im, lm)
        for name, data in WRINKLES.items():
            try:
                wtype = data.get('type', 'curved')
                darker = data.get('darker', False)
                broken = data.get('broken_chance', 0.15)
                width = data.get('width', 3)
                extra = any(k in name.lower() for k in ['under-eye', 'mouth', 'chin'])
                is_fh = (wtype == 'forehead')
                if is_fh and skip_fh:
                    continue

                if wtype == 'forehead':
                    pts = self.generate_forehead_wrinkle_points(lm, data['position'], data['curve'])
                    if pts:
                        self.draw_wrinkle(canvas, pts, darker, broken, width, False, True, landmarks=lm)

                elif wtype == 'radial_grouped':
                    for pr in data['indices']:
                        if len(pr) == 2:
                            i0, i1 = int(pr[0]), int(pr[1])
                            if i0 < len(lm) and i1 < len(lm):
                                pt0 = (int(lm[i0][0]), int(lm[i0][1]))
                                pt1 = (int(lm[i1][0]), int(lm[i1][1]))
                                if self.is_valid_point(pt0, canvas) and self.is_valid_point(pt1, canvas):
                                    self.draw_wrinkle(canvas, [pt0, pt1], darker, broken,
                                                      width, extra, False, landmarks=lm)

                else:
                    indices = data.get('indices', [])
                    if not indices:
                        continue

                    # ---- FIX: handle both flat list and list of pairs ----
                    if isinstance(indices[0], (list, tuple)):
                        # Each element is a pair of landmark indices (short separate wrinkles)
                        for seg in indices:
                            if len(seg) == 2:
                                i0, i1 = int(seg[0]), int(seg[1])
                                if i0 < len(lm) and i1 < len(lm):
                                    pt1, pt2 = lm[i0], lm[i1]
                                    if self.is_valid_point(pt1, canvas) and self.is_valid_point(pt2, canvas):
                                        self.draw_wrinkle(canvas, [pt1, pt2], darker, broken,
                                                      width, extra, False, landmarks=lm)
                    else:
                        # Flat list of integer indices – one continuous line
                        pts = [lm[int(i)] for i in indices if int(i) < len(lm)]
                        pts = [p for p in pts if self.is_valid_point(p, canvas)]
                        if len(pts) >= 2:
                            self.draw_wrinkle(canvas, pts, darker, broken, width, extra, False, landmarks=lm)

            except Exception as e:
                if config.VERBOSE:
                    print(f"  ⚠ {name}: {e}")
        return canvas

    def apply_post_processing(self, im):
        if config.APPLY_BLUR and config.BLUR_AMOUNT > 0:
            # Convert float to int, then ensure odd
            k = int(round(config.BLUR_AMOUNT))
            if k % 2 == 0:
                k += 1
            if k < 1:
                k = 1
            return cv2.GaussianBlur(im, (k, k), 0)
        return im

    def draw_all_wrinkles(self, im, lm, verbose=True, debug_filename=None):
        if im is None or im.size == 0: return im
        im = self.auto_adjust_brightness(im)
        if config.DETECTION_MODE == "hessian":
            canvas, hd = self.draw_hessian_wrinkles(im, lm, debug_filename)
            if not hd:
                if verbose: print("  ⚠ No Hessian — manual")
                canvas = self.draw_manual_wrinkles(im, lm, debug_filename)
                if config.YOUNG_FACE_MODE:
                    canvas = cv2.addWeighted(im, 1 - config.YOUNG_FACE_OPACITY, canvas, config.YOUNG_FACE_OPACITY, 0)
            elif verbose:
                print("  ✓ Hessian drawn")
        elif config.DETECTION_MODE == "hybrid":
            man = self.draw_manual_wrinkles(im, lm, debug_filename)
            hes, hd = self.draw_hessian_wrinkles(im, lm, debug_filename)
            canvas = man if not hd else cv2.addWeighted(man, config.HYBRID_MANUAL_WEIGHT,
                                                        hes, 1 - config.HYBRID_MANUAL_WEIGHT, 0)
            if config.YOUNG_FACE_MODE:
                canvas = cv2.addWeighted(im, 1 - config.YOUNG_FACE_OPACITY, canvas, config.YOUNG_FACE_OPACITY, 0)
        else:
            canvas = self.draw_manual_wrinkles(im, lm, debug_filename)
            if config.YOUNG_FACE_MODE:
                canvas = cv2.addWeighted(im, 1 - config.YOUNG_FACE_OPACITY, canvas, config.YOUNG_FACE_OPACITY, 0)
        eb = self.build_eyebrow_mask(im, lm)
        eb3 = cv2.merge([eb, eb, eb])
        res = np.where(eb3 == 0, im, canvas).astype(np.uint8)
        return self.apply_post_processing(res)

    def close(self):
        if self.hair_segmenter:
            self.hair_segmenter.close()