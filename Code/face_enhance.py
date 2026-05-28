from typing import Optional, List, Tuple
import cv2
import numpy as np
import mediapipe as mp
import config as cfg
from helper_functions import _center_of, _get_points, _local_scale_warp, _local_translate_warp, _make_ellipse_mask, \
    _make_polygon_mask, build_exclusion_mask, pick_mole_position, get_skin_color, draw_mole
from landmarks import (LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR, LEFT_EYEBROW, RIGHT_EYEBROW, LEFT_IRIS,
                       LEFT_UNDER_EYE_LOWER, NOSE_BRIDGE_LEFT, NOSE_BRIDGE_RIGHT, RIGHT_IRIS,
                       LEFT_CHEEK_REF, RIGHT_CHEEK_REF,
                       LEFT_CHEEK_FRONT, LEFT_CHEEK_BACK, RIGHT_CHEEK_BACK, RIGHT_CHEEK_FRONT,
                       LEFT_CHEEK_HIGHLIGHT, RIGHT_CHEEK_HIGHLIGHT,
                       RIGHT_UNDER_EYE_LOWER, FACE_OVAL, LIPS_OUTER, MOLE_ZONES, FRECKLE_REGIONS, SYMMETRIC_PAIRS)
import random
from hair_detection import Hair


class FaceEnhancer:
    """
    Combined facial micro-adjustments with fast landmark detection.
    """

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=cfg.MEDIAPIPE_REFINE_LANDMARKS,
            min_detection_confidence=0.5,
            min_tracking_confidence=cfg.MEDIAPIPE_TRACKING_CONFIDENCE,
        )

        # Frame skipping for live view
        self.frame_counter = 0
        self.last_landmarks = None

        self.hair_detection = Hair()

        # For force detection (photo capture)
        self.face_mesh_full = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,  # Static mode for photos
            max_num_faces=1,
            refine_landmarks=True,  # Full detail for photos
            min_detection_confidence=0.5
        )

    _center_of = staticmethod(_center_of)
    _get_points = staticmethod(_get_points)
    _make_polygon_mask = staticmethod(_make_polygon_mask)
    _make_ellipse_mask = staticmethod(_make_ellipse_mask)
    _local_translate_warp = staticmethod(_local_translate_warp)
    _local_scale_warp = staticmethod(_local_scale_warp)

    def detect_landmarks_fast(self, image: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """
        Fast detection for live view - only returns the 5 landmarks needed for yaw.
        Skips frames for performance.
        """
        self.frame_counter += 1
        if self.frame_counter < cfg.PROCESS_EVERY_N_FRAMES:
            return self.last_landmarks

        self.frame_counter = 0

        try:
            h, w = image.shape[:2]

            # Aggressive downscaling for speed
            target_size = cfg.DETECTION_DOWNSCALE
            if max(h, w) > target_size:
                scale = target_size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                small = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            else:
                small = image
                scale = 1.0

            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)

            if not results.multi_face_landmarks:
                self.last_landmarks = None
                return None

            # Extract ONLY the 5 landmarks needed for yaw calculation
            # Indices: left_eye(133), right_eye(362), nose(1), left_cheek(234), right_cheek(389)
            required_indices = [133, 362, 1, 234, 389]

            landmarks_5 = []
            for idx in required_indices:
                lm = results.multi_face_landmarks[0].landmark[idx]
                landmarks_5.append((int(lm.x * w), int(lm.y * h)))

            self.last_landmarks = landmarks_5
            return landmarks_5

        except Exception as e:
            print(f"Fast detection error: {e}")
            return self.last_landmarks

    def force_detect_full(self, image: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """
        Full 468-point detection for photo capture (slower but accurate).
        """
        try:
            h, w = image.shape[:2]
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh_full.process(rgb)

            if not results.multi_face_landmarks:
                return None

            return [(lm.x * w, lm.y * h)
                    for lm in results.multi_face_landmarks[0].landmark]

        except Exception as e:
            print(f"Full detection error: {e}")
            return None

    def detect_landmarks(self, image: np.ndarray, force: bool = False) -> Optional[List[Tuple[int, int]]]:
        """Legacy method - use detect_landmarks_fast or force_detect_full instead."""
        if force:
            return self.force_detect_full(image)
        return self.detect_landmarks_fast(image)

    def get_last_landmarks(self) -> Optional[List[Tuple[int, int]]]:
        """Get cached landmarks."""
        return self.last_landmarks

    # ================================================================== #
    #  CATEGORY 1: MICRO PROPORTION SHIFTS
    # ================================================================== #
    def _eye_dist(self, lm):
        if len(lm) > 100:  # Full landmarks
            left_center = self._center_of(lm, LEFT_EYE_CONTOUR)
            right_center = self._center_of(lm, RIGHT_EYE_CONTOUR)
        else:  # 5-point landmarks
            left_center = np.array(lm[0])
            right_center = np.array(lm[1])
        return np.linalg.norm(left_center - right_center)

    def adjust_eye_spacing(self, image, lm, strength):
        if len(lm) > 100:
            lc = self._center_of(lm, LEFT_EYE_CONTOUR)
            rc = self._center_of(lm, RIGHT_EYE_CONTOUR)
        else:
            lc = np.array(lm[0])
            rc = np.array(lm[1])
        d = self._eye_dist(lm)
        s, r = d * strength, d * cfg.EYE_SPACING_RADIUS
        result = self._local_translate_warp(image, (int(lc[0]), int(lc[1])), (-s, 0), r)
        return self._local_translate_warp(result, (int(rc[0]), int(rc[1])), (s, 0), r)

    def adjust_eye_size(self, image, lm, scale):
        if len(lm) > 100:
            lc = self._center_of(lm, LEFT_EYE_CONTOUR)
            rc = self._center_of(lm, RIGHT_EYE_CONTOUR)
        else:
            lc = np.array(lm[0])
            rc = np.array(lm[1])
        r = self._eye_dist(lm) * cfg.EYE_SIZE_RADIUS
        result = self._local_scale_warp(image, (int(lc[0]), int(lc[1])), scale, r)
        return self._local_scale_warp(result, (int(rc[0]), int(rc[1])), scale, r)

    def adjust_eyebrow_height(self, image, lm, strength):
        if len(lm) > 100:
            lb = self._center_of(lm, LEFT_EYEBROW)
            rb = self._center_of(lm, RIGHT_EYEBROW)
        else:
            # Approximate from eye positions
            lb = np.array([lm[0][0], lm[0][1] - 30])
            rb = np.array([lm[1][0], lm[1][1] - 30])
        d = self._eye_dist(lm)
        s, r = -d * strength, d * cfg.EYEBROW_RADIUS
        result = self._local_translate_warp(image, (int(lb[0]), int(lb[1])), (0, s), r)
        return self._local_translate_warp(result, (int(rb[0]), int(rb[1])), (0, s), r)

    def adjust_philtrum(self, image, lm, strength):
        if len(lm) > 100:
            lip = np.array(lm[0], dtype=np.float64)
        else:
            lip = np.array(lm[2])  # Nose tip as approximation
        d = self._eye_dist(lm)
        return self._local_translate_warp(image, (int(lip[0]), int(lip[1])),
                                          (0, -d * strength), d * cfg.PHILTRUM_RADIUS)

    def apply_proportion_shifts(self, image, lm, **kwargs):
        r = image.copy()
        if abs(cfg.EYE_SPACING) > 0.001:
            r = self.adjust_eye_spacing(r, lm, cfg.EYE_SPACING)
        if abs(cfg.EYE_SIZE - 1.0) > 0.001:
            r = self.adjust_eye_size(r, lm, cfg.EYE_SIZE)
        if abs(cfg.EYEBROW_HEIGHT) > 0.001:
            r = self.adjust_eyebrow_height(r, lm, cfg.EYEBROW_HEIGHT)
        if abs(cfg.PHILTRUM) > 0.001:
            r = self.adjust_philtrum(r, lm, cfg.PHILTRUM)
        return r

    # ================================================================== #
    #  CATEGORY 3: SKIN TWEEKS - EVEN OUT SKIN TONE
    # ================================================================== #

    def _get_face_mask(self, image_shape, landmarks, fade_dist_px=15, inner_zero_px=5, forehead_boost=0.25,
                       image: np.ndarray = None):
        """
        Create face mask with symmetric forehead expansion.
        forehead_boost = 0.25 expands the top by 25% of eye distance (both left and right).
        """
        h, w = image_shape[:2]
        eye_dist = self._eye_dist(landmarks)

        # --- 1. Get face oval and expand forehead symmetrically ---
        oval_pts = np.array(self._get_points(landmarks, FACE_OVAL), dtype=np.float32)

        # Find centre of face (average of leftmost and rightmost)
        cx = (np.min(oval_pts[:, 0]) + np.max(oval_pts[:, 0])) / 2
        cy = (np.min(oval_pts[:, 1]) + np.max(oval_pts[:, 1])) / 2

        # Identify points above the centre (forehead region)
        # Use a threshold: e.g., y < cy - eye_dist*0.2
        forehead_threshold = cy - eye_dist * 0.2
        forehead_mask = oval_pts[:, 1] < forehead_threshold
        if np.any(forehead_mask):
            # Shift those points upward by forehead_boost * eye_dist
            oval_pts[forehead_mask, 1] -= eye_dist * forehead_boost
            # Keep x coordinates unchanged (symmetric)

        # Ensure we don't go out of bounds
        oval_pts[:, 0] = np.clip(oval_pts[:, 0], 0, w)
        oval_pts[:, 1] = np.clip(oval_pts[:, 1], 0, h)
        oval_pts = oval_pts.astype(np.int32)

        # --- 2. Face oval mask (outer boundary) ---
        face_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(face_mask, [oval_pts], 255)

        # Expand slightly so the mask fully covers face edges
        expand_px = max(5, int(eye_dist * 0.04))
        expand_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_px * 2 + 1, expand_px * 2 + 1))
        face_mask = cv2.dilate(face_mask, expand_k)

        # Tiny blur just to soften the polygon boundary — not enough to fade the edges
        face_mask = cv2.GaussianBlur(face_mask, (7, 7), 2)
        face_mask = face_mask.astype(np.float32) / 255.0

        excl_bin = np.zeros((h, w), dtype=np.uint8)
        for region in [LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR,
                       LEFT_EYEBROW, RIGHT_EYEBROW, LIPS_OUTER]:
            pts = self._get_points(landmarks, region)
            if len(pts) >= 3:
                cv2.fillPoly(excl_bin, [pts], 1)
        excl_bin = cv2.dilate(excl_bin, np.ones((3, 3), np.uint8), iterations=1)

        # --- Add hair mask to exclusion if image provided ---
        if image is not None:
            hair_mask = self.hair_detection.get_hair_mask_sync(image, landmarks=landmarks)
            if hair_mask is not None:
                h_img, w_img = image.shape[:2]
                # Clip beard area — only exclude hair above the nose line
                mid_y = h_img // 2
                hair_mask[mid_y:, :] = 0
                hair_bin = (hair_mask > 0.5).astype(np.uint8)
                excl_bin = cv2.bitwise_or(excl_bin, hair_bin)

        # --- 4. Distance transform and fade-out ---
        dist_map = cv2.distanceTransform(1 - excl_bin, cv2.DIST_L2, 5)

        # Clip dist_map to face area only — prevents image edges from affecting the fade
        dist_map = dist_map * (face_mask > 0.1).astype(np.float32)

        zero_zone = max(0, inner_zero_px)
        fade_zone = max(1, fade_dist_px)
        t = np.clip((dist_map - zero_zone) / fade_zone, 0.0, 1.0)
        excl_weight = t * t * (3.0 - 2.0 * t)

        # --- 5. Combine ---
        final_mask = face_mask * excl_weight

        # Heavy feather at the boundary so it fades smoothly into original skin
        feather_k = max(21, int(eye_dist * 0.15)) * 2 + 1
        final_mask = cv2.GaussianBlur(final_mask, (feather_k, feather_k), feather_k // 4)

        final_mask = np.clip(final_mask * 255, 0, 255).astype(np.uint8)
        return final_mask

    def remove_blemishes(self, frame, strength, face_mask):
        if strength == 0:
            return frame
        k = max(3, int(strength * 0.3) | 1)
        if k % 2 == 0:
            k += 1
        alpha = strength / 100.0
        smooth = cv2.bilateralFilter(frame, k, 75, 75)  # bilateral preserves edges better than gaussian
        result = cv2.addWeighted(frame, 1 - alpha, smooth, alpha, 0)  # no * 0.6 multiplier
        if face_mask is not None:
            mask3 = cv2.cvtColor(face_mask, cv2.COLOR_GRAY2BGR) / 255.0
            result = (result * mask3 + frame * (1 - mask3)).astype(np.uint8)
        return result

    def smooth_skin_new(self, frame, strength, face_mask=None):
        """
        Bilateral filter skin smoothing.
        Ported directly from reference beauty filter.
        strength: 0-100
        """
        if strength == 0:
            return frame
        d = 5
        sigma = int(strength * 0.75) + 1
        smooth = cv2.bilateralFilter(frame, d, sigma, sigma)
        alpha = strength / 100.0
        result = cv2.addWeighted(frame, 1 - alpha, smooth, alpha, 0)
        if face_mask is not None:
            mask3 = cv2.cvtColor(face_mask, cv2.COLOR_GRAY2BGR) / 255.0
            result = (result * mask3 + frame * (1 - mask3)).astype(np.uint8)
        return result

    def even_skin_tone_full(self, frame, strength, face_mask=None, lm=None):
        if strength == 0:
            return frame

        if face_mask is None:
            if lm is None or len(lm) < 100:
                return frame
            face_mask = self._get_face_mask(frame.shape, lm, image=frame)
        else:
            if face_mask.dtype != np.uint8:
                face_mask = (face_mask * 255).astype(np.uint8)

        mask_float = face_mask.astype(np.float32) / 255.0
        total_pixels = max(mask_float.sum(), 1.0)
        alpha = strength / 100.0

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Compute average A and B only (skip L to avoid brightening)
        avg_vals = [None]  # placeholder for ch=0 (not used)
        for ch in range(1, 3):
            avg = np.sum(lab[:, :, ch] * mask_float) / total_pixels
            avg_vals.append(avg)

        # Shift A and B channels toward face average — leave L (lightness) untouched
        for ch in range(1, 3):
            lab[:, :, ch] = np.clip(
                lab[:, :, ch] + (avg_vals[ch] - lab[:, :, ch]) * alpha * mask_float,
                0, 255
            )

        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    def apply_skin_tweaks_new(self, image, lm, **kwargs):
        if len(lm) < 100:
            return image

        face_mask = self._get_face_mask(image.shape, lm, image=image)
        r = image.copy()

        if cfg.SKIN_TONE_STRENGTH_NEW > 0:
            r = self.even_skin_tone_full(r, int(cfg.SKIN_TONE_STRENGTH_NEW * 100), face_mask, lm)
        if cfg.BLEMISH_REMOVAL_STRENGTH > 0:
            r = self.remove_blemishes(r, int(cfg.BLEMISH_REMOVAL_STRENGTH * 100), face_mask)
        if cfg.SMOOTHING_STRENGTH_NEW > 0:
            r = self.smooth_skin_new(r, int(cfg.SMOOTHING_STRENGTH_NEW * 100), face_mask)
        return r

    def whiten_sclera(self, image, lm, **kwargs):
        strength = cfg.SCLERA_STRENGTH

        if len(lm) < 100:
            return image

        result = image.copy()
        ed = self._eye_dist(lm)
        f = max(3, int(ed * 0.04))
        for ec, ii in [(LEFT_EYE_CONTOUR, LEFT_IRIS),
                       (RIGHT_EYE_CONTOUR, RIGHT_IRIS)]:
            em = self._make_polygon_mask(image.shape, self._get_points(lm, ec), feather=f)
            ic = self._center_of(lm, ii)
            ip = self._get_points(lm, ii)
            ir = int(np.max(np.linalg.norm(ip - ic, axis=1)) * 1.2) if len(ip) >= 2 else int(ed * 0.07)
            im = self._make_ellipse_mask(image.shape, (int(ic[0]), int(ic[1])),
                                         (ir, ir), feather=max(2, int(ir * 0.3)))
            scm = np.clip(em - im, 0, 1)
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
            ll = np.sum(lab[:, :, 0] * scm) / max(np.sum(scm), 1.0)
            lab[:, :, 0] = np.clip(lab[:, :, 0] +
                                   max(0, min((cfg.SCLERA_TARGET_L - ll) * strength,
                                              cfg.SCLERA_MAX_BOOST)) * scm, 0, 255)
            for ch in [1, 2]:
                lab[:, :, ch] = np.clip(lab[:, :, ch] -
                                        (lab[:, :, ch] - 128) * scm * strength * cfg.SCLERA_DESAT_RATIO, 0, 255)
            result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        return result

    # ================================================================== #
    #  CATEGORY 3: GEOMETRIC MICRO-EDITS
    # ================================================================== #

    def slim_nose(self, image, lm, strength):
        if len(lm) < 100:
            return image
        d = self._eye_dist(lm)
        s, r = d * strength, d * cfg.NOSE_SLIM_RADIUS
        result = image.copy()
        for idx in NOSE_BRIDGE_LEFT:
            if idx < len(lm):
                pt = lm[idx]
                result = self._local_translate_warp(result, (int(pt[0]), int(pt[1])), (s, 0), r)

        for idx in NOSE_BRIDGE_RIGHT:
            if idx < len(lm):
                pt = lm[idx]
                result = self._local_translate_warp(result, (int(pt[0]), int(pt[1])), (-s, 0), r)
        return result

    def lift_mouth_corners(self, image, lm, strength):
        if len(lm) < 100:
            return image
        d = self._eye_dist(lm)
        s, r = d * strength, d * cfg.MOUTH_LIFT_RADIUS
        result = image.copy()
        for ci in [61, 291]:
            pt = lm[ci]
            result = self._local_translate_warp(result, (int(pt[0]), int(pt[1])), (0, -s), r)
        return result

    def align_pupils(self, image, lm, strength):
        if len(lm) < 100:
            return image, lm
        h, w = image.shape[:2]
        li = self._center_of(lm, LEFT_IRIS)
        ri = self._center_of(lm, RIGHT_IRIS)
        angle = np.degrees(np.arctan2(ri[1] - li[1], ri[0] - li[0]))
        corr = -angle * strength
        if abs(corr) < 0.05:
            return image, lm
        ctr = ((li[0] + ri[0]) / 2, (li[1] + ri[1]) / 2)
        M = cv2.getRotationMatrix2D(ctr, corr, 1.0)
        result = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT)
        updated = [(int((M @ np.array([x, y, 1.0]))[0]),
                    int((M @ np.array([x, y, 1.0]))[1])) for x, y in lm]
        return result, updated

    def apply_geometric_edits(self, image, lm, **kwargs):
        if len(lm) < 100:
            return image
        result = image.copy()
        clm = list(lm)
        if cfg.PUPIL_ALIGN > 0.001:
            result, clm = self.align_pupils(result, clm, cfg.PUPIL_ALIGN)
        if cfg.NOSE_SLIM > 0.001:
            result = self.slim_nose(result, clm, cfg.NOSE_SLIM)
        if cfg.MOUTH_LIFT > 0.001:
            result = self.lift_mouth_corners(result, clm, cfg.MOUTH_LIFT)
        return result

    def widen_or_slim_face(self, image: np.ndarray, lm, strength: float = 0.05, slim=True, **kwargs) -> np.ndarray:
        """
           Opposite of face slimming - pushes cheeks and jaw outward.
           strength: 0.0 = no change, 0.05 = subtle, 0.15 = noticeable, 0.30 = extreme
        """

        h, w = image.shape[:2]
        pts = np.array(lm, dtype=np.float32)
        ed = float(self._eye_dist(lm))

        if ed < 1.0 or abs(strength) < 0.001:
            return image

        # How many pixels beyond the face oval to extend the warp zone.
        # Covers typical hair width so it moves with the cheek.
        HAIR_EXPAND = int(ed * 0.55)

        # ── 1. Centre-line ────────────────────────────────────────────────────
        oval_pts = np.array([lm[i] for i in FACE_OVAL], dtype=np.float32)
        cx = float((oval_pts[:, 0].min() + oval_pts[:, 0].max()) / 2.0)

        # ── 2. Two masks: face oval + expanded warp zone ───────────────────────
        face_oval_pts = self._get_points(lm, FACE_OVAL)

        face_only_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(face_only_mask, [face_oval_pts], 255)

        # Dilate the face mask outward to create the warp zone that
        # catches hair pixels sitting just outside the face oval.
        small_mask = cv2.resize(face_only_mask, (w // cfg.DILATE_SCALE, h // cfg.DILATE_SCALE))
        small_dilated = cv2.dilate(small_mask, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (HAIR_EXPAND // cfg.DILATE_SCALE * 2 + 1, HAIR_EXPAND // cfg.DILATE_SCALE * 2 + 1)))
        warp_zone_mask = cv2.resize(small_dilated, (w, h))

        # ── 3. Per-row face edges (from the warp zone, not just the oval) ──────
        mask_bool = warp_zone_mask > 0
        has_face = mask_bool.any(axis=1)
        left_edge = np.where(has_face, np.argmax(mask_bool, axis=1).astype(np.float32), cx)
        right_edge = np.where(has_face, (w - 1 - np.argmax(mask_bool[:, ::-1], axis=1)).astype(np.float32), cx)

        # Smooth the edge profiles to remove polygon jaggies
        k = max(3, int(ed * 0.12) * 2 + 1)
        left_edge = cv2.GaussianBlur(left_edge.reshape(h, 1), (1, k), 0).reshape(h)
        right_edge = cv2.GaussianBlur(right_edge.reshape(h, 1), (1, k), 0).reshape(h)

        # ── 4. Build flow field with smoothstep ramp ───────────────────────────
        # smoothstep(t) = 3t² - 2t³  maps [0,1]→[0,1] with zero derivative
        # at both ends — no hinge at the face edge, no kink at the nose.
        cols = np.tile(np.arange(w, dtype=np.float32), (h, 1))
        rows_i = np.arange(h, dtype=np.int32)

        le = left_edge[rows_i, None]
        re = right_edge[rows_i, None]

        max_disp_l = (cx - le) * strength
        max_disp_r = (re - cx) * strength

        if not slim:
            max_disp_l = -max_disp_l
            max_disp_r = -max_disp_r

        flow_x = np.zeros((h, w), dtype=np.float32)

        left_mask = cols < cx
        denom_l = np.where(np.abs(cx - le) > 0.5, cx - le, 1.0)
        t_left = np.clip((cx - cols) / np.abs(denom_l), 0.0, 1.0)
        smooth_l = t_left * t_left * (3.0 - 2.0 * t_left)
        flow_x = np.where(left_mask, max_disp_l * smooth_l, flow_x)

        right_mask = cols > cx
        denom_r = np.where(np.abs(re - cx) > 0.5, re - cx, 1.0)
        t_right = np.clip((cols - cx) / np.abs(denom_r), 0.0, 1.0)
        smooth_r = t_right * t_right * (3.0 - 2.0 * t_right)
        flow_x = np.where(right_mask, -max_disp_r * smooth_r, flow_x)

        flow_y = np.zeros((h, w), dtype=np.float32)

        # ── 5. Smoothing ──────────────────────────────────────────────────────
        k_h = max(3, int(ed * 0.04) * 2 + 1)
        flow_x = cv2.GaussianBlur(flow_x, (k_h, 1), 0)

        # ── 6. Mask flow to the warp zone (face + hair) ───────────────────────
        # The warp zone mask is feathered so the displacement fades out
        # gradually at the hair boundary — no hard edge in the warp itself.

        feather_warp_k = max(11, HAIR_EXPAND) * 2 + 1
        feather_k = max(11, int(ed * 0.12)) * 2 + 1

        warp_zone_f = warp_zone_mask.astype(np.float32)  # convert once, reuse below

        warp_feathered = cv2.GaussianBlur(warp_zone_f, (feather_warp_k, feather_warp_k), 0)
        warp_feathered /= max(warp_feathered.max(), 1.0)

        flow_x *= warp_feathered
        flow_y *= warp_feathered

        # ── 7. Remap ──────────────────────────────────────────────────────────
        map_x = (np.arange(w, dtype=np.float32)[None, :] - flow_x).astype(np.float32)
        map_y = (np.arange(h, dtype=np.float32)[:, None] - flow_y).astype(np.float32)

        warped = cv2.remap(image, map_x, map_y,
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)

        # ── 8. Feathered composite ────────────────────────────────────────────
        # Blend uses the warp zone (not just the face oval) so the hair
        # region composites smoothly too.
        face_blend = cv2.GaussianBlur(warp_zone_f, (feather_k, feather_k), 0)
        face_blend /= max(face_blend.max(), 1.0)
        face_blend3 = np.stack([face_blend] * 3, axis=-1)

        result = (warped.astype(np.float32) * face_blend3 +
                  image.astype(np.float32) * (1.0 - face_blend3))

        return np.clip(result, 0, 255).astype(np.uint8)

    def add_random_moles(self, image: np.ndarray, landmarks):
        """Add moles with count determined by MOLE_MAX_COUNT (min always 1)."""
        eye_dist = np.linalg.norm(
            np.mean([landmarks[i] for i in [33, 246, 161, 160, 159, 158, 157, 173, 133]], axis=0) -
            np.mean([landmarks[i] for i in [362, 466, 388, 387, 386, 385, 384, 398, 362]], axis=0))

        max_cnt = int(round(cfg.MOLE_MAX_COUNT))
        if max_cnt < 1:
            return image, []

        # Build distribution list of 5 entries, values from 1 to max_cnt
        # At max_cnt=1 -> [1,1,1,1,1]
        # At max_cnt=2 -> [1,1,1,2,2]  (original)
        # At max_cnt=3 -> [1,2,2,3,3]  (or [1,1,2,3,3] – adjustable)
        # We use linear weighting: lower values more frequent at lower max_cnt.
        possible_counts = []
        for _ in range(5):
            # Triangular distribution favouring lower numbers when max is small
            # Simple: random choice between 1 and max_cnt, but skew
            # Let's create a list manually based on max_cnt
            if max_cnt == 1:
                possible_counts = [1, 1, 1, 1, 1]
                break
            elif max_cnt == 2:
                possible_counts = [1, 1, 1, 2, 2]
                break
            else:  # max_cnt >= 3
                # Build [1,2,2,3,3] as default for max=3
                # For max>3, extend similarly
                possible_counts = [1]
                mid = max_cnt // 2
                for v in range(2, max_cnt + 1):
                    possible_counts.extend([v] * (2 if v < max_cnt else 1))
                # Trim to 5 entries if longer
                possible_counts = possible_counts[:5]
                # Ensure length 5
                while len(possible_counts) < 5:
                    possible_counts.append(max_cnt)
                break

        num_moles = random.choice(possible_counts)

        if num_moles == 0:
            return image, []

        hair_mask = self.hair_detection.get_hair_mask_sync(image, landmarks=landmarks)

        exclusion = build_exclusion_mask(image, landmarks, hair_mask=hair_mask)

        zone_names = list(MOLE_ZONES.keys())
        chosen_zones = random.sample(zone_names, min(num_moles, len(zone_names)))

        result = image.copy()
        placed_positions = []

        for zone_name in chosen_zones:
            zone_idx = MOLE_ZONES[zone_name]
            for _ in range(10):
                pos = pick_mole_position(landmarks, zone_idx)
                if pos is None:
                    break
                cx, cy = pos
                h, w = image.shape[:2]
                if not (0 <= cx < w and 0 <= cy < h) or exclusion[cy, cx] > 0:
                    continue
                too_close = False
                for prev in placed_positions:
                    if np.linalg.norm(np.array(pos) - np.array(prev)) < eye_dist * 0.15:
                        too_close = True
                        break
                if too_close:
                    continue
                skin_color = get_skin_color(image, landmarks, zone_idx)
                result = draw_mole(result, pos, skin_color, eye_dist)
                placed_positions.append(pos)
                break

        return result, placed_positions

    def contour_cheekbones(self, image: np.ndarray, lm, **kwargs) -> np.ndarray:
        if len(lm) < 100:
            return image

        color_strength = cfg.CHEEKBONE_COLOR_STRENGTH
        warp_strength = cfg.CHEEKBONE_WARP_STRENGTH
        h, w = image.shape[:2]
        ed = self._eye_dist(lm)
        feather = max(7, int(ed * cfg.CHEEKBONE_FEATHER_RATIO))

        # ── Yaw estimation (unchanged) ─────────────────────────────────────
        le_pt = np.array(lm[133], dtype=np.float32)
        re_pt = np.array(lm[362], dtype=np.float32)
        lc_pt = np.array(lm[234], dtype=np.float32)
        rc_pt = np.array(lm[454], dtype=np.float32)
        eye_cx = (le_pt[0] + re_pt[0]) / 2.0
        left_dist = eye_cx - lc_pt[0]
        right_dist = rc_pt[0] - eye_cx
        total = left_dist + right_dist
        yaw_fraction = left_dist / max(total, 1.0)

        warp_left = float(np.clip(2.0 * yaw_fraction, 0.3, 1.4)) * warp_strength
        warp_right = float(np.clip(2.0 * (1.0 - yaw_fraction), 0.3, 1.4)) * warp_strength
        color_left = float(np.clip(2.0 * yaw_fraction, 0.3, 1.4)) * color_strength
        color_right = float(np.clip(2.0 * (1.0 - yaw_fraction), 0.3, 1.4)) * color_strength

        NEUTRAL_LEFT = [114, 47, 100, 126, 209]
        NEUTRAL_RIGHT = [343, 277, 329, 355, 429]

        oval_pts = np.array([lm[i] for i in FACE_OVAL], dtype=np.float32)
        face_cx = float((oval_pts[:, 0].min() + oval_pts[:, 0].max()) / 2.0)

        pairs = [
            (LEFT_CHEEK_BACK, LEFT_CHEEK_FRONT, LEFT_CHEEK_HIGHLIGHT, NEUTRAL_LEFT, warp_left, color_left),
            (RIGHT_CHEEK_BACK, RIGHT_CHEEK_FRONT, RIGHT_CHEEK_HIGHLIGHT, NEUTRAL_RIGHT, warp_right, color_right),
        ]

        # ── Build face mask (excludes eyes, eyebrows, lips, and HAIR) ───────
        # Passing `image` triggers hair detection inside `_get_face_mask`
        face_mask = self._get_face_mask(image.shape, lm, image=image)  # uint8, 0-255
        mask_float = face_mask.astype(np.float32) / 255.0
        mask_3ch = np.stack([mask_float] * 3, axis=-1)

        # ── Step 1: Warps (applied to whole image) ─────────────────────────
        result_bgr = image.copy()
        for back_indices, front_indices, _, _, ws, _ in pairs:
            if abs(ws) < 0.001:
                continue
            back_pts = self._get_points(lm, back_indices).astype(np.float32)
            front_pts = self._get_points(lm, front_indices).astype(np.float32)
            all_hollow = np.vstack([back_pts, front_pts])
            hollow_cx = float(np.mean(all_hollow[:, 0]))
            hollow_cy = float(np.mean(all_hollow[:, 1]))
            dx = face_cx - hollow_cx
            push_x = dx * ws * 0.25
            result_bgr = self._local_translate_warp(
                result_bgr, (int(hollow_cx), int(hollow_cy)), (push_x, 0.0), ed * 0.45)

        # ── Blend warped result back to original in hair/eye regions ────────
        result_bgr = (result_bgr.astype(np.float32) * mask_3ch +
                      image.astype(np.float32) * (1.0 - mask_3ch)).astype(np.uint8)

        # ── Step 2: LAB colour work (only on face region) ───────────────────
        lab_full = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_work = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        for back_indices, front_indices, highlight_indices, neutral_indices, _, cs in pairs:
            if abs(cs) < 0.001:
                continue

            # Build cheek masks (same as before)
            shadow_back_mask = self._make_polygon_mask(
                image.shape, self._get_points(lm, back_indices), feather=feather)
            shadow_front_mask = self._make_polygon_mask(
                image.shape, self._get_points(lm, front_indices), feather=feather)
            highlight_mask = self._make_polygon_mask(
                image.shape, self._get_points(lm, highlight_indices), feather=feather)

            # Exclude eye regions (already present in face_mask, but keep for safety)
            for eye_idx in [LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR]:
                eye_ex = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(eye_ex, [self._get_points(lm, eye_idx)], 255)
                eye_ex = cv2.dilate(eye_ex, np.ones((5, 5), np.uint8), iterations=2)
                excl = 1.0 - eye_ex.astype(np.float32) / 255.0
                shadow_back_mask *= excl
                shadow_front_mask *= excl
                highlight_mask *= excl

            # Apply the global face mask (which now also excludes hair)
            shadow_back_mask = np.clip(shadow_back_mask * mask_float, 0.0, 1.0)
            shadow_front_mask = np.clip(shadow_front_mask * mask_float, 0.0, 1.0)
            highlight_mask = np.clip(highlight_mask * mask_float, 0.0, 1.0)

            # Overlap handling (unchanged)
            overlap = np.minimum(shadow_back_mask, highlight_mask)
            shadow_back_mask = np.clip(shadow_back_mask - overlap * 0.7, 0.0, 1.0)
            highlight_mask = np.clip(highlight_mask - overlap * 0.7, 0.0, 1.0)
            shadow_front_mask = np.clip(shadow_front_mask - shadow_back_mask - overlap * 0.5, 0.0, 1.0)

            # Sample skin tone (unchanged)
            skin_bgr = get_skin_color(image, lm, neutral_indices)
            skin_lab = cv2.cvtColor(
                skin_bgr.reshape(1, 1, 3).astype(np.uint8),
                cv2.COLOR_BGR2LAB
            ).reshape(3).astype(np.float32)
            skin_L, skin_a, skin_b = skin_lab

            # Adaptive deltas
            delta_shadow_back = skin_L * cfg.CHEEKBONE_L_DROP * cs
            delta_shadow_front = skin_L * cfg.CHEEKBONE_L_DROP * cs * 0.35
            delta_highlight = skin_L * cfg.CHEEKBONE_L_LIFT * cs * cfg.CHEEKBONE_HIGHLIGHT_RATIO
            warm_shift = float(np.clip(
                (128 - skin_a) * cfg.CHEEKBONE_SHADOW_WARM * 0.04 * cs, 0.0, 4.0))

            # Apply colour adjustments only on face region (hair already excluded by masks)
            lab_work[:, :, 0] = np.clip(
                lab_work[:, :, 0] + delta_highlight * highlight_mask, 0, 255)
            lab_work[:, :, 1] = np.clip(
                lab_work[:, :, 1] + (skin_a - lab_work[:, :, 1]) * 0.12 * cs * highlight_mask, 0, 255)
            lab_work[:, :, 2] = np.clip(
                lab_work[:, :, 2] + (skin_b - lab_work[:, :, 2]) * 0.12 * cs * highlight_mask, 0, 255)

            lab_work[:, :, 0] = np.clip(
                lab_work[:, :, 0] - delta_shadow_back * shadow_back_mask, 0, 255)
            warm_shaped = warm_shift * shadow_back_mask * (1.0 - shadow_back_mask * 0.5)
            lab_work[:, :, 1] = np.clip(
                lab_work[:, :, 1] + warm_shaped, 0, 255)
            b_diff = skin_b - lab_work[:, :, 2]
            b_pull = np.where(b_diff > 0, b_diff * 0.15 * cs, 0.0)
            lab_work[:, :, 2] = np.clip(
                lab_work[:, :, 2] + b_pull * shadow_back_mask, 0, 255)

            lab_work[:, :, 0] = np.clip(
                lab_work[:, :, 0] - delta_shadow_front * shadow_front_mask, 0, 255)

        # ── Final blend: revert hair/eyes back to original LAB ───────────────
        lab_final = (lab_work * mask_float[..., None] +
                     lab_full * (1.0 - mask_float[..., None]))
        return cv2.cvtColor(np.clip(lab_final, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    def remove_eye_bags(self, image, lm, debug=False):
        if len(lm) < 100:
            return image

        strength = cfg.EYE_BAG_REMOVAL_STRENGTH

        result = image.copy()
        ed = self._eye_dist(lm)
        feather = max(5, int(ed * cfg.UNDER_EYE_FEATHER_RATIO))

        debug_mask = np.zeros(image.shape[:2], dtype=np.float32) if debug else None

        # ── Pre-compute brightest cheek reference across both sides ──────────
        cheek_Ls = []
        for cheek_pts in [LEFT_CHEEK_REF, RIGHT_CHEEK_REF]:
            ck_temp = self._make_polygon_mask(image.shape, self._get_points(lm, cheek_pts),
                                              feather=max(3, int(ed * 0.05)))
            lab_temp = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
            cheek_pixels_L = lab_temp[:, :, 0][ck_temp > 0.1]
            skin_pixels = cheek_pixels_L[cheek_pixels_L > 100]
            if len(skin_pixels) > 20:
                cheek_Ls.append(np.percentile(skin_pixels, 60))
            else:
                cheek_Ls.append(np.percentile(cheek_pixels_L, 60))

        for under_pts, cheek_pts, eye_contour in [
            (LEFT_UNDER_EYE_LOWER, LEFT_CHEEK_REF, LEFT_EYE_CONTOUR),
            (RIGHT_UNDER_EYE_LOWER, RIGHT_CHEEK_REF, RIGHT_EYE_CONTOUR),
        ]:
            # ── Mask ─────────────────────────────────────────────────────────
            ec = self._center_of(lm, eye_contour)

            # Estimate head pitch from nose tip vs eye midpoint vertical relationship
            nose_tip = np.array(lm[1])  # landmark 1 = nose tip
            left_eye = np.array(lm[133])  # or use self._center_of(lm, LEFT_EYE_CONTOUR)
            right_eye = np.array(lm[362])
            eye_mid = (left_eye + right_eye) / 2.0

            # When head tilts down, nose_tip.y drops relative to eyes
            # pitch_offset > 0 means head is tilted down
            eye_to_nose_dist = nose_tip[1] - eye_mid[1]
            neutral_ratio = 0.9  # tune this: expected nose-to-eye vertical ratio at neutral
            pitch_offset = max(0, eye_to_nose_dist - ed * neutral_ratio) * 0.4

            pm = self._make_polygon_mask(image.shape, self._get_points(lm, under_pts), feather=feather)

            ellipse_center_y = int(ec[1] + ed * cfg.UNDER_EYE_ELLIPSE_OFFSET_Y + pitch_offset)
            ellipse_center_y = max(ellipse_center_y, int(ec[1] + ed * 0.1))

            em = self._make_ellipse_mask(
                image.shape,
                (int(ec[0]), ellipse_center_y),
                (int(ed * cfg.UNDER_EYE_ELLIPSE_WIDTH), int(ed * cfg.UNDER_EYE_ELLIPSE_HEIGHT)),
                feather=feather,
            )

            cm = np.maximum(pm, em)

            ex = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(ex, [self._get_points(lm, eye_contour)], 255)
            ex = cv2.dilate(ex, np.ones((5, 5), np.uint8), iterations=3)
            cm = np.clip(cm * (1.0 - ex.astype(np.float32) / 255.0), 0.0, 1.0)

            blur_cm = max(11, int(ed * 0.15))
            if blur_cm % 2 == 0:
                blur_cm += 1
            cm = cv2.GaussianBlur(cm, (blur_cm, blur_cm), 0)

            if debug:
                debug_mask = np.maximum(debug_mask, cm)

            # ── Get target skin colour from cheek reference ──────────────────
            target_bgr = get_skin_color(result, lm, cheek_pts)
            target_lab = cv2.cvtColor(np.uint8([[target_bgr]]), cv2.COLOR_BGR2LAB)[0][0].astype(np.float32)

            # Convert current result to LAB once
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)

            # Apply correction to L, a, b using the target skin colour
            for ch, w in [(0, 0.7), (1, 0.4), (2, 0.35)]:  # L gets highest weight
                current = lab[:, :, ch].copy()
                target_val = target_lab[ch]
                delta = (target_val - current) * cm * strength * w
                lab[:, :, ch] = np.clip(current + delta, 0, 255)

            result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

            # ── STEP 2: Graft original texture back onto brightened result ────────
            blur_size = max(7, int(ed * 0.05))
            if blur_size % 2 == 0:
                blur_size += 1

            low_freq_orig = cv2.GaussianBlur(image, (blur_size, blur_size), 0).astype(np.float32)
            high_freq_orig = image.astype(np.float32) - low_freq_orig  # original texture, untouched

            result_low = cv2.GaussianBlur(result, (blur_size, blur_size), 0).astype(np.float32)

            # Brightened shadow layer + original texture = no smoothing artifact
            reconstructed = np.clip(result_low + high_freq_orig, 0, 255).astype(np.uint8)

            # Only swap pixels inside the mask — outside is untouched original
            mask_3ch = np.stack([cm] * 3, axis=-1)
            result = (reconstructed.astype(np.float32) * mask_3ch +
                      result.astype(np.float32) * (1.0 - mask_3ch))
            result = np.clip(result, 0, 255).astype(np.uint8)

        if debug:
            red = np.zeros_like(result)
            red[:, :, 2] = 255
            alpha = debug_mask[..., np.newaxis] * 0.55
            return np.clip(
                result.astype(np.float32) * (1 - alpha) + red.astype(np.float32) * alpha,
                0, 255,
            ).astype(np.uint8)

        return result

    # ================================================================== #
    #  FRECKLE HELPERS  (skin-tone-aware, relative BGR multiplier)
    # ================================================================== #

    @staticmethod
    def _sample_skin_lab(image: np.ndarray, x: int, y: int, radius: int = 4) -> np.ndarray:
        h, w = image.shape[:2]
        x1, y1 = max(0, x - radius), max(0, y - radius)
        x2, y2 = min(w, x + radius + 1), min(h, y + radius + 1)
        patch = image[y1:y2, x1:x2]
        mean_bgr = np.mean(patch.reshape(-1, 3), axis=0).astype(np.uint8)
        lab = cv2.cvtColor(mean_bgr.reshape(1, 1, 3), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
        return lab  # shape (3,)  — L, a, b

    @staticmethod
    def _freckle_color_from_lab(skin_lab: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        L, a, b = skin_lab

        base_bgr = cv2.cvtColor(
            skin_lab.astype(np.uint8).reshape(1, 1, 3),
            cv2.COLOR_LAB2BGR
        )[0, 0].astype(np.float32)

        if L < 80:
            # Deep / Fitzpatrick V–VI
            if rng.random() < 0.35:
                # Ashen variant — slightly lighter, less blue
                scale_b = rng.uniform(0.80, 0.95)
                scale_g = rng.uniform(0.82, 0.95)
                scale_r = rng.uniform(0.85, 0.95)
            else:
                # Warm-dark variant — pull toward brown, not red
                scale_b = rng.uniform(0.50, 0.68)
                scale_g = rng.uniform(0.62, 0.78)
                scale_r = rng.uniform(0.72, 0.88)  # was 0.88–1.05

        elif L < 150:
            # Medium / olive — Fitzpatrick III–IV
            scale_b = rng.uniform(0.58, 0.74)
            scale_g = rng.uniform(0.70, 0.84)
            scale_r = rng.uniform(0.78, 0.90)  # was 0.88–1.05

        else:
            # Light — Fitzpatrick I–II
            scale_b = rng.uniform(0.65, 0.80)
            scale_g = rng.uniform(0.75, 0.88)
            scale_r = rng.uniform(0.82, 0.92)  # was 0.90–1.05

        freckle_bgr = np.array([
            np.clip(base_bgr[0] * scale_b, 0.0, 255.0),
            np.clip(base_bgr[1] * scale_g, 0.0, 255.0),
            np.clip(base_bgr[2] * scale_r, 0.0, 255.0),
        ], dtype=np.float32)

        return freckle_bgr

    @staticmethod
    def _freckle_alpha(skin_L: float, rng: np.random.Generator) -> float:
        if skin_L < 80:
            return float(rng.uniform(0.62, 0.82))
        elif skin_L < 150:
            return float(rng.uniform(0.35, 0.55))  # was 0.38–0.58
        else:
            return float(rng.uniform(0.22, 0.42))  # was 0.28–0.50

    @staticmethod
    def _draw_freckle(canvas: np.ndarray, x: int, y: int,
                      color: np.ndarray, alpha: float,
                      radius: int, rng: np.random.Generator) -> None:
        """
        Draw a single freckle as a Gaussian-feathered irregular ellipse.

        One ellipse mask is drawn hard then Gaussian-blurred so the edge fades
        naturally — identical to a soft brush in photo-editing software.
        This replaces the old two-pass halo/core approach which caused the two
        passes to interfere additively and produce harsh-looking dots.
        """
        angle = float(rng.uniform(0.0, 180.0))
        axes = (radius, max(1, int(radius * rng.uniform(0.50, 1.0))))

        pad = max(axes) + radius + 3
        h_full, w_full = canvas.shape[:2]
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w_full, x + pad + 1), min(h_full, y + pad + 1)
        roi = canvas[y1:y2, x1:x2].astype(np.float32)
        local_cx, local_cy = x - x1, y - y1

        # Hard ellipse → Gaussian blur → soft feathered alpha mask
        mask_u8 = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.ellipse(mask_u8, (local_cx, local_cy),
                    axes, angle, 0, 360, 255, -1, cv2.LINE_AA)

        blur_k = max(3, radius * 2 + 1)
        if blur_k % 2 == 0:
            blur_k += 1
        mask_soft = cv2.GaussianBlur(mask_u8, (blur_k, blur_k), radius * 0.5)
        mask_f = mask_soft.astype(np.float32) / 255.0 * alpha

        color_f = np.array(color, dtype=np.float32)
        for c in range(3):
            roi[:, :, c] += (color_f[c] - roi[:, :, c]) * mask_f

        canvas[y1:y2, x1:x2] = np.clip(roi, 0, 255).astype(np.uint8)

    def add_freckles(self, image: np.ndarray, lm: List[Tuple[int, int]]) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
            Amount of freckles:
            (eye_dist * eye_dist) * density_strength * density_multiplier = Amount
        """

        if len(lm) < 100:
            return image, []

        strength = cfg.FRECKLE_STRENGTH
        density = cfg.FRECKLE_DENSITY_STRENGTH

        h, w = image.shape[:2]
        result = image.copy()
        rng = np.random.default_rng()

        eye_dist = float(self._eye_dist(lm))

        # ── 1. Hair mask & exclusion ──────────────────────────────────────────
        hair_mask = self.hair_detection.get_hair_mask_sync(image, landmarks=lm)
        exclusion_mask = build_exclusion_mask(image, lm, hair_mask=hair_mask)
        exclusion_bin = (exclusion_mask > 0).astype(np.float32)

        # ── 2. Allowed facial regions ─────────────────────────────────────────
        regions = FRECKLE_REGIONS

        full_mask = np.zeros((h, w), dtype=np.uint8)
        for indices in regions.values():
            pts = self._get_points(lm, indices)
            if len(pts) >= 3:
                cv2.fillPoly(full_mask, [pts], 255)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        full_mask = cv2.dilate(full_mask, kernel, iterations=1)
        # Gaussian blur just for smooth mask edges — NOT used as opacity weight
        full_mask_f = cv2.GaussianBlur(full_mask, (9, 9), 3).astype(np.float32) / 255.0
        full_mask_f = full_mask_f * (1.0 - exclusion_bin)

        # ── 3. UV-exposure weight map (peaks at nose bridge / upper cheeks) ───
        nose_tip = np.array(lm[4], dtype=np.float32)

        ys, xs = np.where(full_mask_f > 0.05)
        if len(ys) == 0:
            return image, []

        coords = np.column_stack([xs, ys]).astype(np.float32)
        dist_to_nose = np.linalg.norm(coords - nose_tip, axis=1)
        sigma = eye_dist * 0.8
        uv_weight = np.exp(-0.5 * (dist_to_nose / sigma) ** 2)
        uv_weight *= full_mask_f[ys, xs]
        total = uv_weight.sum()
        if total < 1e-6:
            return image, []
        uv_weight /= total

        # ── 4. Count: scaled to face size, not raw pixel count ────────────────
        # eye_dist² grows with resolution so the same density value produces
        # the same visual result at any image size.
        reference_eye_dist = 180.0  # typical "normal" face size in pixels
        size_factor = reference_eye_dist / max(eye_dist, 50.0)
        size_factor = np.clip(size_factor, 0.5, 2.5)  # don't go crazy in either direction
        n_total = int((eye_dist ** 2) * density * cfg.FRECKLE_DENSITY_MULTIPLIER * size_factor)
        n_total = max(100, min(n_total, 2000))

        # ── 5. 70 % clustered + 30 % scattered placement ─────────────────────
        n_cluster_seeds = max(4, int(n_total * 0.12))
        seed_idx = rng.choice(len(ys), size=n_cluster_seeds,
                              replace=False, p=uv_weight)
        cluster_centers = coords[seed_idx]

        clustered_pts: List[Tuple[int, int]] = []
        per_cluster = max(1, int(n_total * 0.70 / n_cluster_seeds))
        spread = eye_dist * 0.055
        for cx_f, cy_f in cluster_centers:
            offsets = rng.normal(0, spread, (per_cluster, 2))
            for dx, dy in offsets:
                px, py = int(round(cx_f + dx)), int(round(cy_f + dy))
                if 0 <= px < w and 0 <= py < h and full_mask_f[py, px] > 0.05:
                    clustered_pts.append((px, py))

        n_random = max(0, n_total - len(clustered_pts))
        if n_random > 0:
            rand_idx = rng.choice(len(ys), size=min(n_random, len(ys)),
                                  replace=False, p=uv_weight)
            random_pts: List[Tuple[int, int]] = [(int(xs[i]), int(ys[i])) for i in rand_idx]
        else:
            random_pts = []

        all_points = clustered_pts + random_pts

        # ── 6. Adaptive size (larger on darker skin for perceptual weight) ────
        min_radius = max(1, int(eye_dist * 0.008))
        max_radius = max(2, int(eye_dist * 0.022))

        # ── 7. Draw ───────────────────────────────────────────────────────────
        placed_positions: List[Tuple[int, int]] = []
        for (px, py) in all_points:
            if exclusion_mask[py, px] > 0:
                continue
            if full_mask_f[py, px] < 0.05:
                continue

            skin_lab = self._sample_skin_lab(result, px, py)

            # Relative BGR multiplier — works on every skin tone
            freckle_bgr = self._freckle_color_from_lab(skin_lab, rng)

            # Alpha: skin-tone base × global strength ONLY (no mask weight)
            alpha = self._freckle_alpha(skin_lab[0], rng) * strength
            alpha = float(np.clip(alpha, 0.0, 0.85))

            radius = int(rng.integers(min_radius, max_radius + 1))

            # Boost size on darker skin so perceptual weight stays consistent
            size_boost = 1.0 + max(0.0, (128.0 - skin_lab[0]) / 128.0) * 0.60
            radius = max(1, int(round(radius * size_boost)))

            self._draw_freckle(result, px, py, freckle_bgr, alpha, radius, rng)
            placed_positions.append((px, py))

        print(f"[freckles] eye_dist={eye_dist:.1f}  target={n_total}  "
              f"placed={len(placed_positions)}  "
              f"skin_L_sample={self._sample_skin_lab(result, int(nose_tip[0]), int(nose_tip[1]))[0]:.1f}")

        return result, placed_positions

    # ================================================================== #
    #  CATEGORY 4: SYMMETRICAL FACE  (continuous displacement map )
    # ================================================================== #

    # Skip eyes (iris + full eye contour) — warping the eyeball looks wrong.
    # Eyebrows, nose, mouth, cheeks are all moveable.
    _SYMMETRY_SKIP = set(LEFT_EYE_CONTOUR) | set(RIGHT_EYE_CONTOUR) | set(LEFT_EYEBROW) | set(
        RIGHT_EYEBROW) | {
                         468, 469, 470, 471, 472, 473, 474, 475, 476, 477
                     }

    def _fit_midline(self, pts: np.ndarray) -> float:
        left_eye_x = float(np.mean([pts[i][0] for i in [33, 133, 159, 145] if i < len(pts)]))
        right_eye_x = float(np.mean([pts[i][0] for i in [362, 263, 386, 374] if i < len(pts)]))
        return (left_eye_x + right_eye_x) / 2.0

    # ------------------------------------------------------------------
    # Article-style helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_affine_transform(src, src_tri, dst_tri, size):
        warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
        return cv2.warpAffine(src, warp_mat, (size[0], size[1]),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)

    @staticmethod
    def _warp_triangle(img_src, img_dst, t_src, t_dst):
        r_src = cv2.boundingRect(np.float32([t_src]))
        r_dst = cv2.boundingRect(np.float32([t_dst]))

        t_src_rect, t_dst_rect = [], []
        for i in range(3):
            t_src_rect.append(((t_src[i][0] - r_src[0]), (t_src[i][1] - r_src[1])))
            t_dst_rect.append(((t_dst[i][0] - r_dst[0]), (t_dst[i][1] - r_dst[1])))

        mask = np.zeros((r_dst[3], r_dst[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(t_dst_rect), (1.0, 1.0, 1.0), 16, 0)

        img_src_rect = img_src[r_src[1]:r_src[1] + r_src[3], r_src[0]:r_src[0] + r_src[2]]
        size = (r_dst[2], r_dst[3])
        img_dst_rect = FaceEnhancer._apply_affine_transform(img_src_rect, t_src_rect, t_dst_rect, size)
        img_dst_rect = img_dst_rect * mask

        roi = img_dst[r_dst[1]:r_dst[1] + r_dst[3], r_dst[0]:r_dst[0] + r_dst[2]]
        roi *= (1.0, 1.0, 1.0) - mask
        roi += img_dst_rect
        img_dst[r_dst[1]:r_dst[1] + r_dst[3], r_dst[0]:r_dst[0] + r_dst[2]] = roi

    def _build_symmetry_targets(self, pts: np.ndarray, cx: float, blend: float) -> np.ndarray:
        target = pts.copy()

        for (li, ri) in SYMMETRIC_PAIRS:
            if li >= len(pts) or ri >= len(pts) or li == ri:
                continue
            if li in self._SYMMETRY_SKIP or ri in self._SYMMETRY_SKIP:
                continue

            lx, ly = float(pts[li][0]), float(pts[li][1])
            rx, ry = float(pts[ri][0]), float(pts[ri][1])

            # Distance each point sits from the midline
            lx_dist = cx - lx
            rx_dist = rx - cx
            avg_dist = (lx_dist + rx_dist) / 2.0  # equal distance target

            ideal_lx = cx - avg_dist  # place left point symmetrically
            ideal_rx = cx + avg_dist  # place right point symmetrically
            ideal_y = (ly + ry) / 2.0

            target[li, 0] = lx + (ideal_lx - lx) * blend
            target[li, 1] = ly + (ideal_y - ly) * blend
            target[ri, 0] = rx + (ideal_rx - rx) * blend
            target[ri, 1] = ry + (ideal_y - ry) * blend

        return target

    def _build_flow_warp(self, image, src_pts, dst_pts):
        """
        Build a smooth dense displacement map from sparse landmark shifts,
        then remap the image in one shot. Zero triangle seams.
        """
        h, w = image.shape[:2]

        # Compute per-landmark displacement
        displacements = dst_pts - src_pts  # shape (N, 2)

        # Only use points that actually move — ignore static anchors
        moving = np.linalg.norm(displacements, axis=1) > 0.1
        if not np.any(moving):
            return image

        src_moving = src_pts[moving]
        disp_moving = displacements[moving]

        # Build a dense flow field using RBF-style inverse distance weighting
        # For each pixel, interpolate displacement from nearby landmarks
        # We do this efficiently using cv2 with a coarse grid then resize

        # Work on a downscaled grid for speed, then upsample
        scale = 4
        gh, gw = h // scale + 1, w // scale + 1

        grid_x = np.linspace(0, w, gw)
        grid_y = np.linspace(0, h, gh)
        gxx, gyy = np.meshgrid(grid_x, grid_y)  # (gh, gw)

        flow_x = np.zeros((gh, gw), dtype=np.float32)
        flow_y = np.zeros((gh, gw), dtype=np.float32)
        weight_sum = np.zeros((gh, gw), dtype=np.float32)

        # IDW interpolation — each landmark contributes to nearby grid cells
        for idx in range(len(src_moving)):
            lx, ly = src_moving[idx]
            dx, dy = disp_moving[idx]

            # Distance from this landmark to every grid point
            dist2 = (gxx - lx) ** 2 + (gyy - ly) ** 2
            # Gaussian weight — sigma proportional to image size
            sigma = max(h, w) * 0.08
            w_map = np.exp(-dist2 / (2 * sigma ** 2))

            flow_x += w_map * dx
            flow_y += w_map * dy
            weight_sum += w_map

        # Normalize
        w_safe = np.maximum(weight_sum, 1e-6)
        flow_x /= w_safe
        flow_y /= w_safe

        # Upsample flow to full resolution
        flow_x = cv2.resize(flow_x, (w, h), interpolation=cv2.INTER_LINEAR)
        flow_y = cv2.resize(flow_y, (w, h), interpolation=cv2.INTER_LINEAR)

        # Build absolute remap coords
        map_x = (np.arange(w, dtype=np.float32)[None, :] - flow_x).astype(np.float32)
        map_y = (np.arange(h, dtype=np.float32)[:, None] - flow_y).astype(np.float32)

        warped = cv2.remap(image, map_x, map_y,
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)
        return warped

    def apply_symmetry(self, image: np.ndarray, lm) -> np.ndarray:
        h, w = image.shape[:2]
        pts = np.array(lm, dtype=np.float32)
        if pts.ndim == 1 or len(pts) < 100:
            return image

        ed = float(self._eye_dist(lm))
        if ed < 5.0:
            return image

        blend = cfg.SYMMETRY_STRENGTH

        blend = float(np.clip(blend, 0.0, 1.2))
        cx = self._fit_midline(pts)
        src_pts = pts[:468].copy()
        dst_pts = self._build_symmetry_targets(pts, cx, blend)[:468]

        # Flow warp — no triangles, no seams
        warped = self._build_flow_warp(image, src_pts, dst_pts)

        # Oval mask to hide background/hair artifacts
        oval_pts = pts[FACE_OVAL, :2].astype(np.int32)
        hard = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(hard, [oval_pts], 255)

        dist = cv2.distanceTransform(hard, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        feather_px = max(10, int(ed * 0.15))
        alpha = np.clip(dist / feather_px, 0.0, 1.0)

        alpha3 = np.stack([alpha] * 3, axis=-1)
        result = (warped.astype(np.float32) * alpha3 +
                  image.astype(np.float32) * (1.0 - alpha3))
        return np.clip(result, 0, 255).astype(np.uint8)
