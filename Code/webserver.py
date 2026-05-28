import io
import json
import logging
import os
import threading

import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file

from config import ALL_FILTERS, SETTINGS_FILE

app = Flask(__name__)
_display_manager = None
_hardware_manager = None
_app_ref = None

logger = logging.getLogger(__name__)
logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

_disabled_filters = set()

FILTER_META = {
    "Micro_proportion_shifts": {
        "label": "Micro proportion shifts",
        "hint": "Subtle repositioning of eyes, brows & upper lip",
    },
    "Geometric_micro": {
        "label": "Geometric micro edits",
        "hint": "Nose width, mouth corners & eye tilt correction",
    },
    "Skin_color_adjustment": {
        "label": "Skin & colour adjustment",
        "hint": "Tone evening, blemish reduction & smoothing",
    },
    "Whiten_sclera": {
        "label": "Whiten sclera",
        "hint": "Brightens and desaturates the whites of the eyes",
    },
    "Slim_face": {"label": "Slim face", "hint": "Gently narrows the cheeks and jaw"},
    "Widen_face": {"label": "Widen face", "hint": "Gently expands the cheeks outward"},
    "Random_mole": {"label": "Random mole", "hint": "Adds one or more natural-looking moles"},
    "Accentuate_cheekbone": {"label": "Accentuate Cheekbone", "hint": "Contour cheekbones"},
    "Remove_eye_bags": {"label": "Remove eyebags", "hint": "Highlight area under eyes"},
    "Freckles": {"label": "Freckles", "hint": "Add freckles to a face"},
    "Symmetrical_face": {"label": "Symmetry Filter", "hint": "Create a slightly more symmetrical face"},
    "Draw_wrinkles": {"label": "Wrinkles", "hint": "Draw wrinkles"}
}

CATEGORIES = [
    {"id": "skin", "label": "Skin", "keys": ["Skin_color_adjustment", "Whiten_sclera", "Remove_eye_bags"]},
    {"id": "shape", "label": "Shape", "keys": ["Slim_face", "Widen_face", "Accentuate_cheekbone", "Symmetrical_face"]},
    {"id": "face", "label": "Face", "keys": ["Micro_proportion_shifts", "Geometric_micro"]},
    {"id": "extra", "label": "Extra", "keys": ["Random_mole", "Freckles", "Draw_wrinkles"]},
]

# ── Param definitions ─────────────────────────────────────────────────────────
FILTER_PARAMS: dict[str, dict[str, tuple]] = {
    "Micro_proportion_shifts": {
        "EYE_SPACING": (0.0, 0.020, 0.050),
        "EYE_SIZE": (1.0, 1.10, 1.30),
        "EYEBROW_HEIGHT": (0.0, 0.050, 0.120),
        "PHILTRUM": (0.0, 0.030, 0.080),
    },
    "Geometric_micro": {
        "NOSE_SLIM": (0.0, 0.020, 0.055),
        "MOUTH_LIFT": (0.0, 0.012, 0.035),
        "PUPIL_ALIGN": (1.0, 1.00, 1.00),
    },
    "Skin_color_adjustment": {
        "SKIN_TONE_STRENGTH_NEW": (0.0, 0.30, 0.80),
        "BLEMISH_REMOVAL_STRENGTH": (0.0, 0.60, 1.00),
        "SMOOTHING_STRENGTH_NEW": (0.0, 0.40, 0.90),
    },
    "Whiten_sclera": {
        "SCLERA_STRENGTH": (0.0, 0.30, 0.80),
        "SCLERA_MAX_BOOST": (0.0, 50.0, 100.0),
        "SCLERA_DESAT_RATIO": (0.0, 0.5, 1.0),
    },
    "Slim_face": {
        "FACE_SLIM_STRENGTH": (0.0, 0.05, 0.20),
    },
    "Widen_face": {
        "FACE_WIDEN_STRENGTH": (0.0, 0.05, 0.20),
    },
    "Random_mole": {
        "MOLE_MAX_COUNT": (0.0, 2.0, 5.0),
    },
    "Accentuate_cheekbone": {
        "CHEEKBONE_L_DROP": (0.0, 0.12, 0.20),
        "CHEEKBONE_L_LIFT": (0.0, 0.22, 0.35),
        "CHEEKBONE_HIGHLIGHT_RATIO": (1.0, 1.0, 1.2),
        "CHEEKBONE_SHADOW_WARM": (0.0, 2.0, 3.0),
        "CHEEKBONE_COLOR_STRENGTH": (0.0, 0.7, 1.0),
        "CHEEKBONE_WARP_STRENGTH": (0.0, 0.25, 0.35),
    },
    "Remove_eye_bags": {
        "EYE_BAG_REMOVAL_STRENGTH": (0.0, 0.6, 1.0),
    },
    "Freckles": {
        "FRECKLE_STRENGTH": (0.0, 0.6, 1.0),
    },
    "Symmetrical_face": {
        "SYMMETRY_STRENGTH": (0.0, 1, 1.2)
    },
    "Draw_wrinkles": {
        "YOUNG_FACE_OPACITY": (0.0, 0.6, 1.0),
        "HESSIAN_SENSITIVITY": (0.1, 0.45, 0.9),
        "BLUR_AMOUNT": (0.0, 3.0, 9.0),
    },
}


def _true_default_pct(filter_name: str) -> int:
    """
    Compute the true default slider position (0–100) for a filter by averaging
    the linear default positions of all its params.

    For a param with min=0, default=0.3, max=1.0:
      true_pct = (0.3 - 0.0) / (1.0 - 0.0) * 100 = 30

    If min == max (degenerate range), defaults to 50.
    """
    params = FILTER_PARAMS.get(filter_name, {})
    if not params:
        return 50

    pcts = []
    for (min_val, default, max_val) in params.values():
        span = max_val - min_val
        if span == 0:
            pcts.append(50)
        else:
            pcts.append((default - min_val) / span * 100)

    return round(sum(pcts) / len(pcts))


# ── Intensity state ───────────────────────────────────────────────────────────
# Initialise each filter at its true default pct (not a hardcoded 50)
_filter_intensities: dict[str, int] = {
    f: _true_default_pct(f) for f in ALL_FILTERS
}

# ── Preview state ─────────────────────────────────────────────────────────────
_preview_image_path: str | None = None
_preview_lock = threading.Lock()

# Lazy-initialised FaceEnhancer
_face_enhancer = None
_face_enhancer_lock = threading.Lock()

_wrinkles = None
_wrinkles_lock = threading.Lock()


def _get_face_enhancer():
    global _face_enhancer
    with _face_enhancer_lock:
        if _face_enhancer is None:
            try:
                from face_enhance import FaceEnhancer
                _face_enhancer = FaceEnhancer()
            except Exception as e:
                logger.error("Could not load FaceEnhancer: %s", e)
        return _face_enhancer


def _get_wrinkles():
    global _wrinkles
    with _wrinkles_lock:
        if _wrinkles is None:
            try:
                from wrinkles import CombinedWrinkleDrawer
                _wrinkles = CombinedWrinkleDrawer()
            except Exception as e:
                logger.error("Could not load CombinedWrinkleDrawer: %s", e)
        return _wrinkles


PRESETS: list[dict] = [
    {
        "id": "reset",
        "label": "Standard",
        "hint": "All filters at their default values",
        "intensities": {f: _true_default_pct(f) for f in ALL_FILTERS},
    },
    {
        "id": "subtle",
        "label": "Subtle",
        "hint": "Light touch across all filters",
        "intensities": {
            "Micro_proportion_shifts": 25,
            "Geometric_micro": 20,
            "Skin_color_adjustment": 30,
            "Whiten_sclera": 20,
            "Slim_face": 15,
            "Widen_face": 15,
            "Random_mole": 40,
            "Freckles": 30,
            "Remove_eye_bags": 40,
            "Accentuate_cheekbone": 40,
            "Symmetrical_face":  60,
            "Draw_wrinkles": 30,
        },
    },
    {
        "id": "strong",
        "label": "Strong",
        "hint": "Strong effects",
        "intensities": {
            "Micro_proportion_shifts": 80,
            "Geometric_micro": 80,
            "Skin_color_adjustment": 80,
            "Whiten_sclera": 65,
            "Slim_face": 70,
            "Widen_face": 70,
            "Random_mole": 100,
            "Freckles": 80,
            "Remove_eye_bags": 80,
            "Accentuate_cheekbone": 80,
            "Symmetrical_face": 95,
            "Draw_wrinkles": 75,
        },
    },
    {
        "id": "maximum",
        "label": "Maximum",
        "hint": "Everything pushed to the limit",
        "intensities": {f: 100 for f in ALL_FILTERS},
    }
]


def _validate_presets() -> None:
    known = set(ALL_FILTERS)
    for preset in PRESETS:
        orphaned = set(preset["intensities"]) - known
        missing  = known - set(preset["intensities"])
        if orphaned:
            logger.warning("Preset '%s': unknown filter keys: %s", preset['id'], orphaned)
        if missing:
            for f in missing:
                preset["intensities"][f] = _true_default_pct(f)
            logger.info("Preset '%s': filled missing keys with defaults: %s", preset['id'], missing)

_validate_presets()


# ── Settings persistence ──────────────────────────────────────────────────────

def _load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE) as f:
                data = json.load(f)
            for k, v in data.get('intensities', {}).items():
                if k in _filter_intensities:
                    _filter_intensities[k] = max(0, min(100, int(v)))
            for f in data.get('disabled_filters', []):
                if f in ALL_FILTERS:
                    _disabled_filters.add(f)
            global _preview_image_path
            saved_path = data.get('preview_image_path')
            if saved_path and os.path.exists(saved_path):
                _preview_image_path = saved_path
        except Exception as e:
            logger.error("Could not load settings: %s", e)
    _apply_intensities()


def _save_settings():
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump({
                'intensities': dict(_filter_intensities),
                'disabled_filters': list(_disabled_filters),
                'preview_image_path': _preview_image_path,
            }, f, indent=2)
    except Exception as e:
        logger.error("Could not save settings: %s", e)


def _pct_to_value(min_val: float, max_val: float, pct: int) -> float:
    """
    Simple linear interpolation: 0% → min_val, 100% → max_val.
    """
    return min_val + (max_val - min_val) * (pct / 100.0)


def _apply_intensities() -> None:
    import config as cfg
    for filter_name, pct in _filter_intensities.items():
        for attr, (min_val, _default, max_val) in FILTER_PARAMS.get(filter_name, {}).items():
            setattr(cfg, attr, _pct_to_value(min_val, max_val, pct))


# ── Preview rendering ─────────────────────────────────────────────────────────

def _render_preview(source_img: np.ndarray, active_filters: list[str]) -> np.ndarray:
    fe = _get_face_enhancer()
    wrinkles = _get_wrinkles()
    if fe is None or wrinkles is None:
        return source_img

    import config as cfg

    landmarks = fe.force_detect_full(source_img)
    if landmarks is None:
        return source_img

    dispatch = {
        "Micro_proportion_shifts": lambda img, lm: fe.apply_proportion_shifts(img, lm),
        "Geometric_micro":         lambda img, lm: fe.apply_geometric_edits(img, lm),
        "Skin_color_adjustment":   lambda img, lm: fe.apply_skin_tweaks_new(img, lm),
        "Whiten_sclera":           lambda img, lm: fe.whiten_sclera(img, lm),
        "Slim_face":               lambda img, lm: fe.widen_or_slim_face(img, lm, cfg.FACE_SLIM_STRENGTH, slim=True),
        "Widen_face":              lambda img, lm: fe.widen_or_slim_face(img, lm, cfg.FACE_WIDEN_STRENGTH, slim=False),
        "Random_mole":             lambda img, lm: fe.add_random_moles(img, lm)[0],
        "Accentuate_cheekbone":    lambda img, lm: fe.contour_cheekbones(img, lm),
        "Remove_eye_bags":         lambda img, lm: fe.remove_eye_bags(img, lm),
        "Freckles":                lambda img, lm: fe.add_freckles(img, lm)[0],
        "Symmetrical_face":        lambda img, lm: fe.apply_symmetry(img, lm),
        "Draw_wrinkles":           lambda img, lm: wrinkles.draw_all_wrinkles(img, lm),
    }

    result = source_img.copy()
    for f in active_filters:
        if _filter_intensities.get(f, 0) == 0:
            continue
        fn = dispatch.get(f)
        if fn is None:
            continue
        try:
            result = fn(result, landmarks)
        except Exception as e:
            logger.error("Filter '%s' failed: %s", f, e)

    return result


def _load_preview_source() -> np.ndarray | None:
    with _preview_lock:
        path = _preview_image_path

    if path and os.path.exists(path):
        img = cv2.imread(path)
        if img is not None:
            h, w = img.shape[:2]
            if w > 720:
                scale = 720 / w
                img = cv2.resize(img, (720, int(h * scale)), interpolation=cv2.INTER_AREA)
        return img
    return None


def _img_to_jpeg_bytes(img: np.ndarray, quality: int = 88) -> bytes:
    success, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


# ── Server init ───────────────────────────────────────────────────────────────

def init_webserver(display, hardware, app_ref=None):
    global _display_manager, _hardware_manager, _app_ref
    _display_manager = display
    _hardware_manager = hardware
    _app_ref = app_ref
    _load_settings()
    display.webserver_block = False


def start_webserver(display, hardware, port=5000, app_ref=None):
    init_webserver(display, hardware, app_ref=app_ref)
    thread = threading.Thread(
        target=lambda: app.run(host='0.0.0.0', port=port, debug=False, threaded=True),
        daemon=True
    )
    thread.start()
    logger.info("Webserver running on http://0.0.0.0:%s", port)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/tune')
def tune():
    return render_template('tune.html')


# ── Filter on/off ─────────────────────────────────────────────────────────────

@app.route('/filters', methods=['GET'])
def get_filters_state():
    return jsonify({
        "filters": [
            {"name": f, "enabled": f not in _disabled_filters}
            for f in ALL_FILTERS
        ]
    })


@app.route('/filters/<name>/toggle', methods=['POST'])
def toggle_filter(name):
    if name not in ALL_FILTERS:
        return jsonify({"error": "Unknown filter"}), 404
    if name in _disabled_filters:
        _disabled_filters.discard(name)
        enabled = True
    else:
        _disabled_filters.add(name)
        enabled = False
    _save_settings()
    return jsonify({"name": name, "enabled": enabled})


# ── Filter intensities ────────────────────────────────────────────────────────

@app.route('/filter-intensities', methods=['GET'])
def get_intensities():
    return jsonify({"intensities": dict(_filter_intensities)})


@app.route('/filter-intensities', methods=['POST'])
def set_intensities():
    data = request.get_json(force=True)
    incoming = data.get("intensities", {})
    for name, pct in incoming.items():
        if name in _filter_intensities:
            _filter_intensities[name] = max(0, min(100, int(pct)))
    _apply_intensities()
    _save_settings()
    return jsonify({"ok": True, "intensities": dict(_filter_intensities)})


# ── Preview endpoints ─────────────────────────────────────────────────────────

@app.route('/preview/upload', methods=['POST'])
def upload_preview_image():
    global _preview_image_path

    if 'image' not in request.files:
        return jsonify({"error": "No image field"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    save_dir = os.path.dirname(SETTINGS_FILE)
    os.makedirs(save_dir, exist_ok=True)
    dest = os.path.join(save_dir, 'preview_source.jpg')

    img_bytes = file.read()
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Could not decode image"}), 400

    cv2.imwrite(dest, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    with _preview_lock:
        _preview_image_path = dest

    _save_settings()
    return jsonify({"ok": True, "path": dest})


@app.route('/preview/has-image', methods=['GET'])
def preview_has_image():
    src = _load_preview_source()
    return jsonify({"has_image": src is not None})


@app.route('/preview/original', methods=['GET'])
def preview_original():
    src = _load_preview_source()
    if src is None:
        return jsonify({"error": "No preview image set"}), 404
    return send_file(
        io.BytesIO(_img_to_jpeg_bytes(src)),
        mimetype='image/jpeg',
        max_age=0
    )


@app.route('/preview/render', methods=['GET'])
def preview_render():
    src = _load_preview_source()
    if src is None:
        return jsonify({"error": "No preview image set"}), 404

    override = request.args.get('filters')
    if override:
        active = [f.strip() for f in override.split(',') if f.strip() in ALL_FILTERS]
    else:
        active = []

    try:
        result = _render_preview(src, active)
    except Exception as e:
        logger.error("Render error: %s", e)
        return jsonify({"error": str(e)}), 500

    return send_file(
        io.BytesIO(_img_to_jpeg_bytes(result)),
        mimetype='image/jpeg',
        max_age=0
    )


@app.route('/presets', methods=['GET'])
def get_presets():
    return jsonify({"presets": PRESETS})


@app.route('/presets/<preset_id>/apply', methods=['POST'])
def apply_preset(preset_id):
    preset = next((p for p in PRESETS if p['id'] == preset_id), None)
    if preset is None:
        return jsonify({"error": "Unknown preset"}), 404

    for name, pct in preset['intensities'].items():
        if name not in _filter_intensities:
            continue
        _filter_intensities[name] = max(0, min(100, int(pct)))

    _apply_intensities()
    _save_settings()
    return jsonify({"ok": True, "preset": preset_id, "intensities": dict(_filter_intensities)})


@app.route('/filter-meta', methods=['GET'])
def get_filter_meta():
    """
    Returns everything the frontend needs to build itself, including
    the true default pct for each filter so the UI can render the
    default marker at the correct position.
    """
    filters_out = []
    for name in ALL_FILTERS:
        if name not in FILTER_META:
            continue

        default_pct = _true_default_pct(name)

        filters_out.append({
            "name": name,
            "label": FILTER_META[name]["label"],
            "hint": FILTER_META[name]["hint"],
            "intensity": _filter_intensities.get(name, default_pct),
            "enabled": name not in _disabled_filters,
            "default_pct": default_pct,  # ← true linear default position
        })

    return jsonify({
        "filters": filters_out,
        "categories": CATEGORIES,
    })