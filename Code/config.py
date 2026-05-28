import os

# Base directory = wherever this config file lives
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Settings file
SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")

# Image paths
def _img(filename):
    return os.path.join(BASE_DIR, filename)

IMAGE_PATHS = {
    "Original":                  _img("original.png"),
    "Geometric_micro":           _img("geometric_micro.jpg"),
    "Micro_proportion_shifts":   _img("micro_proportion_shifts.jpg"),
    "Skin_color_adjustment":     _img("skin_color_adjustment.jpg"),
    "Symmetrical_face":          _img("symmetrical_face.jpg"),
    "Whiten_sclera":             _img("whiten_sclera.jpg"),
    "Slim_face":                 _img("slim_face.jpg"),
    "Widen_face":                _img("widen_face.jpg"),
    "Random_mole":               _img("random_mole.jpg"),
    "Draw_wrinkles":             _img("draw_wrinkles.jpg"),
    "Accentuate_cheekbone":      _img("accentuate_cheekbone.jpg"),
    "Remove_eye_bags":           _img("remove_eye_bags.jpg"),
    "Freckles":                  _img("freckles.jpg"),
}

DIFF_PATHS = {
    "Micro_proportion_shifts":   _img("diff_proportion.png"),
    "Geometric_micro":           _img("diff_geometric.png"),
    "Skin_color_adjustment":     _img("diff_skin.png"),
    "Symmetrical_face":          _img("diff_symmetry.png"),
    "Whiten_sclera":             _img("diff_white_sclera.png"),
    "Slim_face":                 _img("diff_slim.png"),
    "Widen_face":                _img("diff_widen.jpg"),
    "Random_mole":               _img("diff_mole.jpg"),
    "Draw_wrinkles":             _img("diff_wrinkles.jpg"),
    "Accentuate_cheekbone":      _img("diff_accentuate_cheekbone.jpg"),
    "Remove_eye_bags":           _img("diff_eye_bags.png"),
    "Freckles":                  _img("diff_freckles.jpg"),
}

ALL_FILTERS = [
    "Micro_proportion_shifts",
    "Geometric_micro",
    "Skin_color_adjustment",
    "Whiten_sclera",
    "Slim_face",
    "Widen_face",
    "Random_mole",
    "Accentuate_cheekbone",
    "Remove_eye_bags",
    "Freckles",
    "Symmetrical_face",
    "Draw_wrinkles"
]

ORIGINAL_KEY = "Original"


# ================= GPIO =================
GPIO_CHIP = 4
BUTTON1 = 15 #1
BUTTON2 = 9 #2
BUTTON3 = 10 #3
BUTTON4 = 11 #4

# ================= Display =================

button_to_index = {
    BUTTON1: 0,
    BUTTON2: 1,
    BUTTON3: 2,
    BUTTON4: 3,
}


SCREEN_W, SCREEN_H = 1080, 1920  # ✅ portrait

MONITOR_POSITIONS = [
    (1721,    0),   # DVI-I-1   (leftmost)
    (2801, 0),   # DVI-I-2
    (3881, 0),   # HDMI-A-1
    (4961, 0),   # HDMI-A-2  (rightmost)
]


# ====================================================================== #
#  FACE ENHANCER FILTER CONFIGURATION
# ====================================================================== #

# ---- Category 1: Proportion Shifts ----
EYE_SPACING = 0.020
EYE_SIZE = 1.1
EYEBROW_HEIGHT = 0.05
PHILTRUM = 0.03
EYE_SPACING_RADIUS = 0.8
EYE_SIZE_RADIUS = 0.35
EYEBROW_RADIUS = 0.4
PHILTRUM_RADIUS = 0.35

# ---- Category 2: Skin & Color Tweaks ----
UNDER_EYE_FEATHER_RATIO = 0.22
UNDER_EYE_ELLIPSE_OFFSET_Y = 0.28
UNDER_EYE_ELLIPSE_WIDTH = 0.24
UNDER_EYE_ELLIPSE_HEIGHT = 0.14

EYE_BAG_REMOVAL_STRENGTH = 0.6

# ---- Category 3: Geometric Micro-Edits ----
PUPIL_ALIGN = 1.0
NOSE_SLIM = 0.020
MOUTH_LIFT = 0.012
NOSE_SLIM_RADIUS = 0.25
MOUTH_LIFT_RADIUS = 0.2


# ---- Category 4: Symmetry Edits ----
SYMMETRY_STRENGTH = 1

# ---- Category 5: Face Widen/Slim ----
FACE_WIDEN_STRENGTH = 0.05
FACE_SLIM_STRENGTH = 0.05  # Strength of face slimming effect (0.0 - 1.0)
DILATE_SCALE = 4

# ---- Category 6: Accentuate Cheekbone ----
CHEEKBONE_L_DROP = 0.12  # was 0.22, much lighter shadow
CHEEKBONE_L_LIFT = 0.22  # was 0.14, stronger highlight
CHEEKBONE_HIGHLIGHT_RATIO = 1.0  # was 0.7, full strength
CHEEKBONE_SHADOW_WARM = 2.0  # was 3.0
CHEEKBONE_COLOR_STRENGTH = 0.7  # 0.0 = off, 1.0 = dramatic
CHEEKBONE_WARP_STRENGTH = 0.25
CHEEKBONE_FEATHER_RATIO = 0.18  # blur size relative to eye distance

# ---- Category 7: Smooth Skin Edits ----
BLEMISH_REMOVAL_STRENGTH = 0.6
SMOOTHING_STRENGTH_NEW = 0.4
SKIN_TONE_STRENGTH_NEW = 0.3

# ---- Category 8: Moles  ----
MOLE_MAX_COUNT = 2.0

# ---- Category 9: Freckles ----
FRECKLE_DENSITY_STRENGTH = 0.002
FRECKLE_DENSITY_MULTIPLIER = 3
FRECKLE_STRENGTH = 0.6


# ---- Category 10: Whiten sclera ----
SCLERA_STRENGTH = 0.3
SCLERA_TARGET_L = 220
SCLERA_MAX_BOOST = 50
SCLERA_DESAT_RATIO = 0.5


# ---- Category 11: Wrinkles ----

# Wrinkle detection mode
# "manual" = landmark-based only
# "hessian" = automated detection from image data
# "hybrid" = blend of both methods
DETECTION_MODE = "hessian"


# Hybrid blend weight
HYBRID_MANUAL_WEIGHT = 0.4

# Young face mode
YOUNG_FACE_MODE = True
YOUNG_FACE_OPACITY = 0.6

# Hessian detection parameters
HESSIAN_SCALE_MIN = 0.5
HESSIAN_SCALE_MAX = 4.0
HESSIAN_SCALE_STEP = 0.5
HESSIAN_SENSITIVITY = 0.45

# Manual wrinkle parameters
BLUR_AMOUNT = 3
APPLY_BLUR = True

APPLY_HAIR_DETECTION = True

# Output verbosity
VERBOSE = True

# Region-specific styles for Hessian-detected wrinkles
HESSIAN_REGION_STYLES = {
    'forehead': {'darker': False, 'broken_chance': 0.1, 'width': 4},
    'under_eye_left': {'darker': True, 'broken_chance': 0.15, 'width': 3},
    'under_eye_right': {'darker': True, 'broken_chance': 0.15, 'width': 3},
    'crows_feet_left': {'darker': False, 'broken_chance': 0.25, 'width': 2},
    'crows_feet_right': {'darker': False, 'broken_chance': 0.25, 'width': 2},
    'nasolabial_left': {'darker': False, 'broken_chance': 0.1, 'width': 3},
    'nasolabial_right': {'darker': False, 'broken_chance': 0.1, 'width': 3},
    'marionette_left': {'darker': False, 'broken_chance': 0.15, 'width': 3},
    'marionette_right': {'darker': False, 'broken_chance': 0.15, 'width': 3},
    'other': {'darker': False, 'broken_chance': 0.2, 'width': 2}
}

# ====================================================================== #
#  FACE ENHANCER CONFIGURATION
# ====================================================================== #

# ---- Performance Tuning ----
PROCESS_EVERY_N_FRAMES = 8  # Process 1 out of 8 frames for live view
DETECTION_DOWNSCALE = 256   # Downscale to 256px for faster detection
MEDIAPIPE_REFINE_LANDMARKS = False  # Disable iris landmarks for speed
MEDIAPIPE_TRACKING_CONFIDENCE = 0.3


# ====================================================================== #
#  HELPER FUNCTIONS CONFIGURATIONS
# ====================================================================== #
# ---- Dark Image Auto-Adjustment ----
DARK_IMAGE_THRESHOLD = 120
TARGET_LUMINANCE = 130
GAMMA_MIN = 0.6
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)
CLAHE_BLEND_WEIGHT = 0.3

# ====================================================================== #
#  ALIGNMENT GUIDE CONFIGURATIONS
# ====================================================================== #
YAW_THRESHOLD = 0.35
STABLE_FRAMES_REQUIRED = 8
OVAL_CENTER_Y_RATIO = 0.42
OVAL_RX_RATIO = 0.30   # 1200 * 0.30 = 360px
OVAL_RY_RATIO = 0.30   # 360px = perfect circle
COLOR_GOOD = (80, 220, 80)
COLOR_BAD = (40, 40, 220)
COLOR_NO_FACE = (255, 100, 0)
LEFT_EYE_COLOR = (255, 0, 0)
RIGHT_EYE_COLOR = (255, 0, 0)
NOSE_COLOR = (0, 0, 255)
CHEEK_COLOR = (0, 255, 255)
LANDMARK_RADIUS = 8
EXPECTED_EYE_CHEEK_RATIO = 0.46
OVAL_THICKNESS = 6
OVAL_FILL_ALPHA = 0.08
FONT_SCALE_MSG = 1.5
FONT_THICKNESS = 3
HAIR_THRESHOLD=0.35

# ====================================================================== #
# HANDLE FLOW CONFIGURATIONS
# ====================================================================== #
FILTER_START = 0.3
FILTER_END = 0.9