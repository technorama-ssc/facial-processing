LEFT_EYE_CONTOUR = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE_CONTOUR = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
             377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
NOSE_TIP_AREA = [1, 2, 98, 327, 168, 6, 197, 195, 5, 4, 19, 94]

LEFT_UNDER_EYE_LOWER = [111, 117, 118, 119, 120, 121, 128, 245, 193]
RIGHT_UNDER_EYE_LOWER = [340, 346, 347, 348, 349, 350, 357, 465, 417]

NOSE_BRIDGE_LEFT = [193, 122, 196, 3, 51, 45, 44, 1]
NOSE_BRIDGE_RIGHT = [417, 351, 419, 248, 281, 275, 274, 1]

LEFT_CHEEK_REF  = [174, 188, 128, 121, 120, 119, 229, 228, 31, 226, 35, 143, 116, 111, 117, 118, 101, 126, 217]
RIGHT_CHEEK_REF = [412, 399, 437, 355, 330, 347, 346, 345, 372, 446, 261, 448, 449, 348, 350, 357]

# MediaPipe landmark indices used for yaw estimation
LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362

NOSTRIL_AREA = [1, 45, 115, 219, 2, 294, 344, 275]

FRECKLE_REGIONS = {
    "forehead": [71, 68, 103, 67, 109, 10, 338, 297, 332, 284, 251,
                 301, 291, 334, 296, 336, 8, 107, 66, 105, 62],
    "cheek_nose": [34, 234, 227, 123, 147, 215, 207, 206, 203, 220,
                   45, 4, 275, 440, 344, 423, 426, 427, 411, 376,
                   352, 447, 454, 372, 264, 448, 449, 450, 451, 452,
                   453, 465, 351, 6, 122, 245, 232, 231, 230, 229,
                   228, 31, 35],
    "lower_cheek_left": [58, 138, 172, 135, 136, 169, 202, 57, 186,
                         92, 165, 216, 192],
    "lower_cheek_right": [288, 416, 436, 319, 287, 430, 365, 364,
                          397, 367],
}


LEFT_CHEEK_HIGHLIGHT  = [116, 117, 118, 36, 50, 123, 116]
RIGHT_CHEEK_HIGHLIGHT = [345, 346, 347, 266, 352, 345]

LEFT_CHEEK_BACK   = [132, 177, 147, 137, 205]
RIGHT_CHEEK_BACK  = [361, 401, 376, 366, 425]

LEFT_CHEEK_FRONT  = [205, 206, 207, 187]
RIGHT_CHEEK_FRONT = [425, 426, 427, 411]

REGIONS = {
    # Eye regions
    "Right Under-Eye": {
        "indices": [357, 350, 349, 348, 449, 448, 261, 446, 255, 254, 253, 252, 256, 341],
        "color": (255, 150, 50),  # Blue
    },
    "Left Under-Eye": {
        "indices": [128, 121, 120, 119, 229, 228, 31, 226, 25, 110, 24, 23, 22, 26, 233],
        "color": (255, 150, 50),  # Blue (same as right)
    },
    "Right Over-Eye": {
        "indices": [465, 413, 441, 442, 443, 283, 300, 383, 353, 342, 260, 259, 257, 258, 286, 464],
        "color": (255, 100, 150),  # Purple-blue
    },
    "Left Over-Eye": {
        "indices": [245, 243, 190, 56, 28, 27, 29, 30, 113, 124, 156, 70, 53, 222, 221, 189],
        "color": (255, 100, 150),  # Purple-blue (same)
    },
    "Right Eyebrow": {
        "indices": [285, 336, 296, 334, 293, 300, 276, 283, 282, 295],
        "color": (50, 200, 255),  # Orange-yellow
    },
    "Left Eyebrow": {
        "indices": [55, 65, 52, 53, 46, 70, 63, 105, 66, 107],
        "color": (50, 200, 255),  # Orange-yellow (same)
    },
    "Left Sclera (approx)": {
        "indices": LEFT_EYE_CONTOUR,
        "color": (0, 255, 0)
    },
    "Right Sclera (approx)": {
        "indices": RIGHT_EYE_CONTOUR,
        "color": (0, 255, 0)
    },

    # Forehead regions
    "Right Forehead": {
        "indices": [9, 151, 10, 338, 297, 332, 284, 251, 301, 276, 293, 334, 296, 336],
        "color": (180, 130, 255),  # Pink
    },
    "Left Forehead": {
        "indices": [9, 151, 10, 109, 67, 103, 54, 21, 71, 70, 63, 105, 66, 107],
        "color": (180, 130, 255),  # Pink (same)
    },

    # Cheek regions
    "Right Upper Cheek": {
        "indices": [412, 399, 437, 355, 330, 347, 346, 345, 372, 446, 261, 448, 449, 348, 350, 357],
        "color": (100, 255, 150),  # Lime green
    },
    "Left Upper Cheek": {
        "indices": [174, 188, 128, 121, 120, 119, 229, 228, 31, 226, 35, 143, 116, 111, 117, 118, 101, 126, 217],
        "color": (100, 255, 150),  # Lime green (same)
    },
    "Right Middle Cheek": {
        "indices": [345, 346, 347, 330, 355, 429, 331, 425, 352],
        "color": (50, 220, 100),  # Green
    },
    "Left Middle Cheek": {
        "indices": [116, 111, 117, 118, 101, 126, 209, 203, 205, 123],
        "color": (50, 220, 100),  # Green (same)
    },
    "Right Lower Cheek": {
        "indices": [352, 376, 433, 416, 436, 426, 331, 425],
        "color": (0, 180, 80),  # Dark green
    },
    "Left Lower Cheek": {
        "indices": [203, 206, 216, 192, 213, 147, 123, 205],
        "color": (0, 180, 80),  # Dark green (same)
    },

    # Nose areas
    "Nose Tip Lower": {
        "indices": [1, 275, 440, 344, 360, 363, 281, 5, 51, 134, 131, 115, 220, 45],
        "color": (0, 200, 200),  # Yellow
    },
    "Nose Tip Upper": {
        "indices": [131, 134, 51, 5, 281, 363, 360, 420, 456, 248, 195, 3, 236, 198],
        "color": (0, 180, 220),  # Warm yellow
    },
    "Nose Tip Combined": {
        "indices": [1, 275, 440, 344, 360, 420, 456, 248, 195, 3, 236, 198, 131, 115, 220, 45],
        "color": (0, 220, 240),  # Bright yellow
    },
    "Nose Bridge": {
        "indices": [236, 3, 195, 248, 456, 419, 351, 417, 9, 193, 122, 196],
        "color": (0, 150, 255),  # Orange
    },
    "Eyebrow-Nose Right": {
        "indices": [9, 336, 417],
        "color": (100, 150, 255),  # Light orange
    },
    "Eyebrow-Nose Left": {
        "indices": [9, 107, 193],
        "color": (100, 150, 255),  # Light orange (same)
    },

    # Mouth areas
    "Philtrum Area": {
        "indices": [186, 92, 165, 167, 164, 393, 391, 410, 294, 289, 2, 75, 219],
        "color": (150, 100, 200),  # Mauve
    },
    "Right Chin": {
        "indices": [152, 175, 199, 200, 18, 313, 406, 335, 273, 287, 432, 434, 367, 397, 365, 379, 378, 377],
        "color": (200, 150, 100),  # Tan
    },
    "Left Chin": {
        "indices": [152, 175, 199, 200, 18, 83, 182, 106, 43, 57, 212, 214, 138, 172, 136, 150, 149, 176, 148],
        "color": (200, 150, 100),  # Tan (same)
    },
    "Lips Outline": {
        "indices": [0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37],
        "color": (60, 60, 230),  # Red
    },
    "Upper Lip": {
        "indices": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 311, 13, 82, 81, 80, 191, 78],
        "color": (80, 80, 255),  # Bright red
    },
    "Lower Lip": {
        "indices": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 324, 318, 402, 317, 14, 87, 178, 88, 95],
        "color": (50, 50, 200),  # Dark red
    },
    "Sclera_Left": {
        "indices": LEFT_EYE_CONTOUR,
        "color": (255, 255, 255),
    },
    "Sclera_Right": {
        "indices": RIGHT_EYE_CONTOUR,
        "color": (255, 255, 255),
    },
}

MOLE_ZONES = {
    "left_cheek":  [123, 50, 36, 137, 205, 206, 177, 147, 187],
    "right_cheek": [352, 280, 266, 366, 425, 426, 401, 376, 411],
    "chin_left":   [32, 201, 200, 175, 152],
    "chin_right":  [262, 421, 418, 396, 152],
    "forehead_left":  [71, 68, 104, 69, 108],
    "forehead_right": [301, 298, 333, 299, 337],
    "nose_left":   [219, 218, 237, 44, 1],
    "nose_right":  [439, 438, 457, 274, 1],
    "upper_lip":   [164, 167, 165, 92, 186, 57, 43],
    "lower_lip":   [377, 396, 394, 322, 410, 287, 273],
}

WRINKLES = {
    # Under-eye wrinkles (darker)
    "Right Inner Under-Eye-Wrinkles": {
        "indices": [256, 252, 253, 254, 339],
        "type": "curved",
        "darker": True,
        "broken_chance": 0.15,
        "width": 3.5
    },
    "Left Inner Under-Eye-Wrinkles": {
        "indices": [26, 22, 23, 24, 110],
        "type": "curved",
        "darker": True,
        "broken_chance": 0.15,
        "width": 3.5
    },
    "Right Outer Under-Eye-Wrinkles": {
        "indices": [452, 451, 450, 449, 448],
        "type": "curved",
        "darker": True,
        "broken_chance": 0.15,
        "width": 3.5
    },
    "Left Outer Under-Eye-Wrinkles": {
        "indices": [232, 231, 230, 229, 228],
        "type": "curved",
        "darker": True,
        "broken_chance": 0.15,
        "width": 3.5
    },

    # Side-eye wrinkles
    "Right Side-Eye-Wrinkles": {
        "indices": [[446, 353], [446, 265], [446, 261]],
        "type": "radial_grouped",
        "darker": False,
        "broken_chance": 0.25,
        "width": 2
    },
    "Left Side-Eye-Wrinkles": {
        "indices": [[226, 124], [226, 35], [226, 31]],
        "type": "radial_grouped",
        "darker": False,
        "broken_chance": 0.25,
        "width": 2
    },

    # Forehead wrinkles
    "Forehead Wrinkle Bottom": {
        "type": "forehead",
        "position": 0.25,
        "curve": -0.3,
        "darker": False,
        "broken_chance": 0.1,
        "width": 4
    },
    "Forehead Wrinkle Middle": {
        "type": "forehead",
        "position": 0.45,
        "curve": -0.2,
        "darker": False,
        "broken_chance": 0.1,
        "width": 4
    },
    "Forehead Wrinkle Top": {
        "type": "forehead",
        "position": 0.65,
        "curve": -0.15,
        "darker": False,
        "broken_chance": 0.1,
        "width": 4
    },

    # Mouth wrinkles
    "Right Inner Mouth Corner-Wrinkle": {
        "indices": [410, 287, 273],
        "type": "curved",
        "darker": False,
        "broken_chance": 0.1,
        "width": 5
    },
    "Left Inner Mouth Corner-Wrinkle": {
        "indices": [186, 57, 43],
        "type": "curved",
        "darker": False,
        "broken_chance": 0.1,
        "width": 5
    },
    "Right Outer Mouth Corner-Wrinkle": {
        "indices": [426, 436, 432, 422],
        "type": "curved",
        "darker": False,
        "broken_chance": 0.1,
        "width": 4.5
    },
    "Left Outer Mouth Corner-Wrinkle": {
        "indices": [206, 216, 212, 202],
        "type": "curved",
        "darker": False,
        "broken_chance": 0.1,
        "width": 4.5
    },

    # Nose wrinkles
    "Nose Right Side-Wrinkles": {
        "indices": [357, 399, 456],
        "type": "curved",
        "darker": False,
        "broken_chance": 0.2,
        "width": 3
    },
    "Nose Left Side-Wrinkles": {
        "indices": [128, 174, 236],
        "type": "curved",
        "darker": False,
        "broken_chance": 0.2,
        "width": 3
    },

    # Chin wrinkle
    "Chin wrinkle": {
        "indices": [32, 201, 200, 421, 262],
        "type": "curved",
        "darker": False,
        "broken_chance": 0.15,
        "width": 3
    },
}


SYMMETRIC_PAIRS = [
    # Face oval / jaw
    (10, 338), (21, 251), (54, 284), (103, 332), (67, 297),
    # Eyebrows
    (55, 285), (52, 282), (53, 283), (46, 276), (105, 334),
    (66, 296), (107, 336), (70, 300), (63, 293),
    # Eyes (contour)
    (33, 263), (7, 249), (163, 382), (144, 381), (145, 374),
    (153, 373), (154, 390), (155, 249), (133, 362), (246, 466),
    (161, 388), (160, 387), (159, 386), (158, 385), (157, 384), (173, 398),
    # Iris
    (468, 473), (469, 474), (470, 475), (471, 476), (472, 477),
    # Nose (midline – keep only one side, e.g., (1,1) but better to handle separately)
    # For midline, use a separate list if needed.
    # Nose sides
    (45, 275), (115, 344), (122, 351), (131, 360), (134, 363),
    (193, 417), (196, 419), (198, 420), (220, 440), (236, 456),
    (3, 248), (51, 281),
    # Cheeks
    (101, 330), (111, 340), (116, 345), (117, 346), (118, 347),
    (123, 352), (126, 355), (143, 372), (147, 376), (174, 399),
    (188, 412), (192, 416), (205, 425), (206, 426), (209, 429),
    (213, 433), (216, 436), (217, 437),
    # Mouth outer
    (61, 291), (76, 306), (77, 307), (78, 308), (80, 310),
    (81, 311), (82, 312), (84, 314), (87, 317), (88, 318),
    (91, 321), (92, 322), (95, 324),
    # Mouth inner
    (37, 267), (39, 269), (40, 270), (75, 305), (146, 375),
    (165, 391), (167, 393), (178, 402), (181, 405), (182, 406),
    (185, 409), (186, 410), (191, 415), (219, 439),
    # Lower face / chin
    (43, 273), (57, 287), (106, 335), (136, 365), (138, 367),
    (148, 377), (149, 378), (150, 379), (172, 397), (176, 400),
    (212, 432), (214, 434),
]