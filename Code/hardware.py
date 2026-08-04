import time
import threading
import cv2
import lgpio
import logging
import subprocess
import signal
import sys
from config import (
    BUTTON1, BUTTON2, BUTTON3, BUTTON4,
    GPIO_CHIP, SCREEN_W, SCREEN_H, IMAGE_PATHS
)

logging.basicConfig(level=logging.WARNING)


def _is_stream_running():
    """Check if camera stream is already running"""
    try:
        result = subprocess.run(['pgrep', '-f', 'rpicam-vid'],
                                capture_output=True)
        return result.returncode == 0
    except:
        return False


def _resize_to_fit_screen(frame):
    if frame is None:
        return None

    h, w = frame.shape[:2]

    # Scale so height fills the full portrait screen
    scale = SCREEN_H / h
    new_w = int(w * scale)
    new_h = SCREEN_H

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Center-crop width to fit portrait screen
    x_offset = (new_w - SCREEN_W) // 2
    cropped = resized[:, x_offset:x_offset + SCREEN_W]

    return cropped


class HardwareManager:

    def __init__(self):
        self.running = True
        self.stream_process = None
        self.cap = None

        # Set up signal handler for clean shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # --- Start camera stream to v4l2loopback ---
        self._start_camera_stream()

        # --- Open camera via v4l2loopback device ---
        self.cap = None

        # Wait for camera device to be ready
        time.sleep(2)

        # Try to open /dev/video10 (v4l2loopback device)
        for dev in [10, 0, 2, 4]:
            try:
                logging.info(f"Trying /dev/video{dev}...")
                cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)

                if cap.isOpened():
                    # Use camera's native resolution (640x480) or set to a standard 4:3 resolution
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1536)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 864)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    # Test read
                    ret, test_frame = cap.read()
                    if ret:
                        self.cap = cap
                        logging.info(f"✓ Camera opened on /dev/video{dev} at 640x480")
                        break
                    else:
                        cap.release()
                        logging.warning(f"✗ /dev/video{dev} opened but read failed")
                else:
                    logging.warning(f"✗ /dev/video{dev} not opened")

            except Exception as e:
                logging.warning(f"Error with /dev/video{dev}: {e}")

        if not self.cap or not self.cap.isOpened():
            logging.error("Failed to open camera")
            raise RuntimeError("Failed to open camera on any video device")

        self._init_gpio()

        self._prev_button_states = dict.fromkeys([BUTTON1, BUTTON2, BUTTON3, BUTTON4], 1)

        # Double buffer
        self.frame_buffers = [None, None]
        self.read_buffer_idx = 0
        self.write_buffer_idx = 1
        self.buffer_lock = threading.Lock()

        # Start frame capture thread
        threading.Thread(
            target=self._capture_frames,
            daemon=True
        ).start()

        threading.Thread(
            target=self._watchdog,
            daemon=True
        ).start()

    def _watchdog(self):
        """Restart camera stream if it dies"""
        consecutive_failures = 0

        while self.running:
            time.sleep(10)

            # Check if pipeline is alive
            if not _is_stream_running():
                logging.warning("Watchdog: camera stream died — restarting...")
                subprocess.run(['sudo', 'pkill', '-f', 'ffmpeg'], capture_output=True)
                time.sleep(1)
                self._start_camera_stream()
                time.sleep(3)
                consecutive_failures += 1
                continue

            # Check if frames are actually coming through
            with self.buffer_lock:
                frame = self.frame_buffers[self.read_buffer_idx]

            if frame is None:
                consecutive_failures += 1
                logging.warning(f"Watchdog: no frames (failure #{consecutive_failures})")

                if consecutive_failures >= 3:
                    logging.warning("Watchdog: restarting full pipeline...")
                    subprocess.run(['sudo', 'pkill', '-f', 'rpicam-vid'], capture_output=True)
                    subprocess.run(['sudo', 'pkill', '-f', 'ffmpeg'], capture_output=True)
                    time.sleep(2)
                    self._start_camera_stream()
                    time.sleep(3)
                    consecutive_failures = 0
            else:
                consecutive_failures = 0

    def _signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully"""
        logging.info("Received interrupt signal, cleaning up...")
        self.cleanup()
        sys.exit(0)

    def _start_camera_stream(self):
        try:
            # Always kill existing first
            subprocess.run(['sudo', 'pkill', '-f', 'rpicam-vid'], capture_output=True)
            subprocess.run(['sudo', 'pkill', '-f', 'ffmpeg'], capture_output=True)
            time.sleep(1)

            logging.info("Starting camera stream at 1536x864...")
            cmd = 'rpicam-vid -t 0 --width 1536 --height 864 --autofocus-mode continuous --autofocus-range normal --codec yuv420 --framerate 20 --output - 2>/dev/null | ffmpeg -f rawvideo -pix_fmt yuv420p -s 1536x864 -r 20 -i - -f v4l2 -pix_fmt yuyv422 /dev/video10 2>/dev/null &'
            self.stream_process = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL,
                                                   stderr=subprocess.DEVNULL)
            time.sleep(2)
            logging.info("Camera stream started")

        except Exception as e:
            logging.warning(f"Could not start camera stream: {e}")

    def _capture_frames(self):
        frame_count = 0
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    if frame_count % 30 == 0:
                        logging.warning("Camera read failed, retrying...")
                    time.sleep(0.01)
                    frame_count += 1
                    continue

                frame_count = 0

                # Resize frame to fit screen while maintaining aspect ratio
                display_frame = _resize_to_fit_screen(frame)

                with self.buffer_lock:
                    self.frame_buffers[self.write_buffer_idx] = display_frame
                    self.read_buffer_idx, self.write_buffer_idx = (
                        self.write_buffer_idx, self.read_buffer_idx
                    )
            except Exception as e:
                logging.error(f"Frame capture error: {e}")
                time.sleep(0.1)

    def get_frame(self):
        with self.buffer_lock:
            frame = self.frame_buffers[self.read_buffer_idx]
            return frame.copy() if frame is not None else None

    def capture_photos(self):
        frame = self.get_frame()
        if frame is not None:
            frame = cv2.flip(frame, 1)
            save_path = IMAGE_PATHS["Original"]
            cv2.imwrite(save_path, frame)
            return save_path
        else:
            logging.error("Photo capture failed — no frame available")
            return None

    def _init_gpio(self):
        self.h = lgpio.gpiochip_open(GPIO_CHIP)

        for pin in [BUTTON1, BUTTON2, BUTTON3, BUTTON4]:
            try:
                lgpio.gpio_free(self.h, pin)
            except Exception:
                pass

        time.sleep(0.1)

        for pin in [BUTTON1, BUTTON2, BUTTON3, BUTTON4]:
            lgpio.gpio_claim_input(self.h, pin, lgpio.SET_PULL_UP)

    def check_buttons(self):
        just_pressed = []

        for btn in [BUTTON1, BUTTON2, BUTTON3, BUTTON4]:
            state = lgpio.gpio_read(self.h, btn)
            prev = self._prev_button_states[btn]

            if prev == 0 and state == 1:
                just_pressed.append(btn)

            self._prev_button_states[btn] = state

        return just_pressed

    def cleanup(self):
        """Cleanup all resources - called on exit"""
        logging.info("Hardware cleanup started")
        self.running = False
        time.sleep(0.2)

        # Release OpenCV camera
        if self.cap:
            self.cap.release()
            logging.info("✓ OpenCV camera released")

        # Kill camera stream process
        if self.stream_process:
            self.stream_process.terminate()
            logging.info("✓ Camera stream terminated")

        # Kill any remaining rpicam-vid processes
        subprocess.run(['sudo', 'pkill', '-f', 'rpicam-vid'], capture_output=True)
        subprocess.run(['sudo', 'pkill', '-f', 'ffmpeg'], capture_output=True)

        # Close GPIO
        lgpio.gpiochip_close(self.h)
        cv2.destroyAllWindows()
        logging.info("✓ Hardware cleanup complete")