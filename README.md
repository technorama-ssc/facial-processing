# Technorama — Facial Processing

An interactive exhibition installation running on a Raspberry Pi 5 with multiple screens. 
Visitors stand in front of a camera, align their face, and the system secretly applies facial enhancements to their photo. 
The photo is then shown alongside three "impostor" images on four screens and visitors have to guess which one is the real unedited them.


---

## How It Works

1. **Live camera view** — User sees themselves on screen with an oval guide overlay
2. **Alignment detection** — MediaPipe detects face landmarks; the system waits until the face is centered, straight,
   and hair is out of the way
3. **Capture & filter** — once aligned for 3 seconds, a photo is taken and 3 of 12 available filters are applied
   automatically
4. **Grid reveal** — all 4 screens show one image each (original + 3 filtered versions, shuffled)
5. **Button selection** — User presses one of 4 physical buttons to guess which image is the original one
6. **Result & diff overlay** — the correct image is revealed, along with a colored overlay showing exactly what was
   changed

---

## Hardware

| Component      | Details                                                    |
|----------------|------------------------------------------------------------|
| Raspberry Pi 5 | Main compute                                               |
| Camera         | Raspberry Pi Camera Module via `rpicam-vid`                |
| Displays       | 4 × portrait monitors (1080×1920) via DisplayLink USB dock |
| Buttons        | 4 physical GPIO buttons (pins 15, 9, 10, 11)               |
| USB dock       | Club3D USB-A 3.0 Dual 4K dock (DisplayLink)                |

---

## Project Structure

```
facial_processing/
├── Code/
│   ├── static/                 
│   │   ├── index.css           # CSS Styling for index.html 
│   │   └── tune.css            # CSS Styling for tune.html 
│   ├── templates/
│   │   ├── index.html          # Web UI: filter on/off toggles
│   │   └── tune.html           # Web UI: per-filter intensity sliders
│   ├── main.py                 # Entry point — app lifecycle, main loop, state machine
│   ├── handle_flow.py          # State handlers
│   ├── alignment_guide.py      # Face alignment overlay, yaw estimation, hair check
│   ├── face_enhance.py         # Face Filters
│   ├── display.py              # DisplayManager — 4-screen window management
│   ├── hardware.py             # Camera stream, GPIO buttons, frame capture
│   ├── config.py               # All tunable constants (thresholds, filter strengths, paths)
│   ├── landmarks.py            # MediaPipe landmark index maps for all facial regions
│   ├── hair_detection.py       # MediaPipe hair segmentation
│   ├── hessian_detector.py     # Frangi vesselness filter for automatic wrinkle detection
│   ├── helper_functions.py     # Filter pipeline, warp utilities, diff overlay
│   ├── utils.py                # Text rendering, Bezier curves, image fitting helpers
│   ├── wrinkles.py             # Wrinkle Filter
│   └── webserver.py            # Flask admin UI — toggle filters, set intensities, preview
└── Config/
    ├── fail_save_monitoring.sh # Watchdog wrapper — restarts main.py, camera etc. on crash
    ├── display_setup.sh        # Docking Station Connection Setup
    └── setup.sh                # Package installation, Service-File creation

```

---

## Setup

You can find everything that you need to set up this project in the [Tutorial](./Documentation/Tutorial.md)

---

## The 12 Filters

| Filter                    | What it does                                         |
|---------------------------|------------------------------------------------------|
| `Micro_proportion_shifts` | Subtle repositioning of eyes, brows, and upper lip   |
| `Geometric_micro`         | Nose width, mouth corner lift, pupil tilt correction |
| `Skin_color_adjustment`   | Skin tone evening, blemish reduction, smoothing      |
| `Whiten_sclera`           | Brightens and desaturates the whites of the eyes     |
| `Slim_face`               | Narrows cheeks and jaw                               |
| `Widen_face`              | Expands cheeks outward                               |
| `Symmetrical_face`        | Warps face toward bilateral symmetry                 |
| `Accentuate_cheekbone`    | Shadow/highlight contouring on cheekbones            |
| `Remove_eye_bags`         | Brightens under-eye area toward cheek skin tone      |
| `Random_mole`             | Adds 1–2 natural-looking moles                       |
| `Freckles`                | Adds freckles                                        |
| `Draw_wrinkles`           | Hessian-detected or landmark-based wrinkle drawing   |

---

## What You Can Tune

### Alignment

| Setting                  | Default Value | Description                                                    | Tuning                                                                                                     |
|--------------------------|---------------|----------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| `YAW_THRESHOLD`          | `0.35`        | How straight the face must be before capture triggers.         | Lower → stricter \| Higher → more lenient                                                                  |
| `STABLE_FRAMES_REQUIRED` | `8`           | How many consecutive aligned frames before the photo is taken. | Lower → captures faster, more risk of catching a blink/micro-movement \| Higher → more reliable but slower |
| `HAIR_THRESHOLD`         | `0.35`        | Maximum fraction of the face oval that can be covered by hair. | Lower → blocks capture if even a little hair is in the oval \| Higher → allows fringes/bangs through       |
| `OVAL_RX_RATIO`          | `0.30`        | Oval width as fraction of the frame's shorter axis             | Lower → smaller oval, face must be closer to camera \| Higher → larger oval, more generous fit zone        |
| `OVAL_RY_RATIO`          | `0.30`        | Oval height as fraction of the frame's shorter axis            | Lower → smaller oval, face must be closer to camera \| Higher → larger oval, more generous fit zone        |
---

## Web Admin UI

A Flask server runs on port `5000`. Access it from any device on the same network:

```
http://<raspberry-pi-ip>:5000
```

- **`/`** — Toggle individual filters on/off (minimum 3 must stay active)
- **`/tune`** — Per-filter intensity sliders with live preview (upload a test photo)
- Presets available: Standard / Subtle / Strong / Maximum
- Settings auto-save to `Config/settings.json`

---

## Key Dependencies

| Package                 | Purpose                                            |
|-------------------------|----------------------------------------------------|
| `mediapipe`             | Face mesh (468 landmarks), hair segmentation       |
| `opencv-python`         | Camera capture, image processing, display          |
| `numpy`                 | Array operations for all warp/mask math            |
| `lgpio`                 | GPIO button reading                                |
| `flask`                 | Web admin interface                                |
| `scipy`                 | Gaussian derivatives for Hessian wrinkle detection |
| `Pillow`                | Unicode text rendering (German umlauts)            |
| `rpicam-vid` + `ffmpeg` | Camera → v4l2loopback pipeline                     |

---