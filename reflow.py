#!/usr/bin/env python3
'''
Les Wright 21 June 2023
https://youtube.com/leslaboratory
A Python program to read, parse and display thermal data from the Topdon TC001 Thermal camera!

macOS adaptation: Uses ffmpeg pipe instead of cv2.VideoCapture for reliable
avfoundation capture with raw YUYV data (preserves thermal data).
'''
import os
import time
import argparse
import subprocess
import numpy as np
import cv2
import logging
import sys
import threading
from typing import List, Tuple, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class CameraParameters(object):
    def __init__(self, frameWidth=256, frameHeight=384, fps=25, pixelFormat="yuyv422"):
        self.frameWidth: int = frameWidth
        self.frameHeight: int = frameHeight
        self.frameSize: int = self.frameWidth * self.frameHeight * 2
        self.fps: int = fps
        self.pixelFormat: str = pixelFormat


class CameraFeed(object):
    def __init__(self, cameraParameters):
        self.cameraParameters: CameraParameters = cameraParameters
        self.imdata: np.ndarray = None
        self.thdata: np.ndarray = None

    def open(self, device=0):
        ffmpeg_cmd = [
            "ffmpeg",
            "-f", "avfoundation",
            "-framerate", str(self.cameraParameters.fps),
            "-video_size", f"{self.cameraParameters.frameWidth}x{self.cameraParameters.frameHeight}",
            "-pixel_format", self.cameraParameters.pixelFormat,
            "-i", str(device),
            "-pix_fmt", self.cameraParameters.pixelFormat,
            "-f", "rawvideo",
            "-"
        ]
        self.pipe: subprocess.Popen = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=10**8
        )

    def close(self):
        self.pipe.terminate()

    def readFrame(self) -> Tuple[np.ndarray, np.ndarray]:
        # Read a raw YUYV422 frame from the ffmpeg pipe
        raw = self.pipe.stdout.read(self.cameraParameters.frameSize)
        if len(raw) != self.cameraParameters.frameSize:
            raise ValueError("Failed to read frame from ffmpeg pipe.")

        # Reshape to (height, width, 2) — same as cv2.VideoCapture with CAP_PROP_CONVERT_RGB=0
        frame = np.frombuffer(raw, np.uint8).reshape(
            (self.cameraParameters.frameHeight, self.cameraParameters.frameWidth, 2))
        imdata, thdata = np.array_split(frame, 2)
        self.imdata = imdata
        self.thdata = thdata
        return (self.imdata, self.thdata)


class ThermalData(object):

    def __init__(self, cameraFeed: CameraFeed):
        self.cameraFeed: CameraFeed = cameraFeed
        self.imdata: np.ndarray = cameraFeed.imdata
        self.thdata: np.ndarray = cameraFeed.thdata
        self.temperature_data: np.ndarray = None
        self.max_temperature: float = None
        self.min_temperature: float = None
        self.average_temperature: float = None

    def convert_to_celsium(self, rawtemp) -> float:
        return (rawtemp/64)-273.15

    def extract_temperatures_fast(
        self, roi: Optional[Tuple[int, int, int, int]] = None
    ) -> Tuple[float, float, float]:
        """Vectorized temperature extraction: returns (avg, max, center) in one shot.

        Args:
            roi: Optional (x1, y1, x2, y2) in sensor pixel coordinates.
                 If provided, only the pixels within this rectangle are used
                 for avg/max/min/center calculations.  The full frame is still
                 read (needed for the heatmap), but statistics are computed on
                 the sub-region only.
        """
        imdata, thdata = self.cameraFeed.readFrame()
        self.imdata = imdata
        self.thdata = thdata
        hi = thdata[..., 0].astype(np.float64)
        lo = thdata[..., 1].astype(np.float64) * 256
        temp_grid = (hi + lo) / 64.0 - 273.15

        if roi is not None:
            x1, y1, x2, y2 = roi
            region = temp_grid[y1:y2, x1:x2]
        else:
            region = temp_grid

        self.average_temperature = float(region.mean())
        self.max_temperature = float(region.max())
        self.min_temperature = float(region.min())
        cy, cx = region.shape[0] // 2, region.shape[1] // 2
        center_temp = float(region[cy, cx])
        return self.average_temperature, self.max_temperature, center_temp

    def extract_temperature_data(self) -> None:
        imdata, thdata = self.cameraFeed.readFrame()
        self.imdata = imdata
        self.thdata = thdata
        self.temperature_data = np.zeros(
            (self.thdata.shape[0], self.thdata.shape[1]))
        for row in range(self.thdata.shape[0]):
            for col in range(self.thdata.shape[1]):
                pixel = self.thdata[row, col]
                hi = pixel[0]
                lo = pixel[1]
                lo = lo.astype(np.uint16)*256
                rawtemp = hi+lo
                temp = self.convert_to_celsium(rawtemp)
                self.temperature_data[row, col] = temp
        self.max_temperature = np.max(self.temperature_data)
        self.min_temperature = np.min(self.temperature_data)
        self.average_temperature = np.mean(self.temperature_data)


class ThermalView(object):
    def __init__(self, thermalData: ThermalData):
        self.thermalData: ThermalData = thermalData
        self.scale = 3  # scale multiplier
        self.alpha = 1.0  # Contrast control (1.0-3.0)
        self.colormap = 0
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.dispFullscreen = False
        self.rad = 0  # blur radius
        self.threshold = 2
        self.hud = True
        self.recording = False
        self.elapsed = "00:00:00"
        self.snaptime = "None"

    def show(self) -> None:
        imdata = self.thermalData.imdata
        # Convert the real image to RGB
        bgr = cv2.cvtColor(imdata,  cv2.COLOR_YUV2BGR_YUYV)
        # Contrast
        bgr = cv2.convertScaleAbs(bgr, alpha=self.alpha)  # Contrast
        # bicubic interpolate, upscale and blur
        print(self.windowWidth, self.windowHeight)
        bgr = cv2.resize(bgr, (self.windowWidth, self.windowHeight),
                         interpolation=cv2.INTER_CUBIC)  # Scale up!
        if self.rad > 0:
            bgr = cv2.blur(bgr, (self.rad, self.rad))

        # apply colormap
        heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_JET)

        cv2.imshow('Thermal Output', heatmap)
        keyPress = cv2.waitKey(1)

        if keyPress == ord('q'):
            self.close()
            self.thermalData.cameraFeed.close()
            sys.exit(0)

    def init_window(self) -> None:
        self.windowWidth = self.thermalData.thdata.shape[1]*self.scale
        self.windowHeight = self.thermalData.thdata.shape[0]*self.scale
        cv2.namedWindow('Thermal', cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow('Thermal', self.windowWidth, self.windowHeight)
        cv2.setWindowProperty(
            'Thermal', cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_GUI_NORMAL)

    def close(self):
        cv2.destroyAllWindows()


def sac305_profile(t: np.ndarray) -> np.ndarray:
    """
    SAC305 No-Clean Solder Paste reflow profile with smooth curves.

    Returns target temperature (°C) for each time point t (seconds).

    Uses piecewise smooth functions (exponential approach, logarithmic soak,
    parabolic peak, exponential decay) instead of linear interpolation so the
    curve matches the shape shown in the Chip Quik SMD291SNL datasheet rather
    than having sharp corners at each waypoint.

    Profile per Chip Quik SMD291SNL datasheet (Sn96.5/Ag3.0/Cu0.5):
      - Preheat ramp:   25°C → 150°C   over   0–90s   (exponential approach)
      - Thermal soak:  150°C → 175°C   over  90–180s  (logarithmic, gentle)
      - Ramp to reflow: 175°C → 217°C  over 180–210s  (exponential approach)
      - Reflow / TAL:  217°C → 249°C → 217°C  210–270s  (parabolic peak)
      - Cooling:       217°C →  25°C   over 270–370s  (exponential decay)

    Ref: https://www.chipquik.com/datasheets/SMD291SNL.pdf
    """
    t = np.asarray(t, dtype=np.float64)
    result = np.full_like(t, 25.0)

    def _exp_ease(p, k=3.0):
        """Normalised exponential blend mapping p:[0,1] -> [0,1].

        Concave-down (fast initial change that slows toward the end),
        modelling an exponential approach to equilibrium.
        Exact endpoints: _exp_ease(0)=0, _exp_ease(1)=1.
        Larger k = more pronounced curvature.
        """
        return (1.0 - np.exp(-k * p)) / (1.0 - np.exp(-k))

    # --- Preheat ramp: 0–90s, 25 → 150°C ---
    # Exponential approach: heats quickly at first (large delta-T with oven),
    # then rate decreases as board approaches soak temperature.
    m = (t >= 0) & (t < 90)
    result[m] = 25.0 + 125.0 * _exp_ease(t[m] / 90.0, k=2.5)

    # --- Thermal soak: 90–180s, 150 → 175°C ---
    # Gentle, logarithmic-like rise (small k ≈ nearly linear with slight curve).
    m = (t >= 90) & (t < 180)
    result[m] = 150.0 + 25.0 * _exp_ease((t[m] - 90.0) / 90.0, k=1.2)

    # --- Ramp to reflow: 180–210s, 175 → 217°C ---
    # Logarithmic approach (concave-up: starts slow, accelerates) with a
    # steeper rate than the soak phase.  Models the board lagging behind a
    # sharply rising oven setpoint, then catching up toward liquidus.
    m = (t >= 180) & (t < 210)
    p = (t[m] - 180.0) / 30.0
    # log1p-based blend: 0→1 concave-up.  np.log1p(e-1)=1, so the
    # normalisation gives exact endpoints.
    result[m] = 175.0 + 42.0 * np.log1p(p * (np.e - 1.0))

    # --- Reflow / TAL: 210–270s, parabolic peak at t=240, T=249°C ---
    # Smooth bell: rises from 217 to 249 at t=240 and returns to 217.
    m = (t >= 210) & (t < 270)
    result[m] = 249.0 - 32.0 * ((t[m] - 240.0) / 30.0) ** 2

    # --- Cooling: 270–370s, 217 → 25°C ---
    # Exponential decay: fast initial drop that slows as board approaches
    # ambient temperature.
    m = (t >= 270) & (t <= 370)
    p = (t[m] - 270.0) / 100.0
    result[m] = 25.0 + 192.0 * (1.0 - _exp_ease(p, k=3.0))

    # Beyond the profile: ambient
    result[t > 370] = 25.0

    return result


def plot_sac305_profile():
    """Plot the SAC305 No-Clean Solder Paste reflow profile with annotated zones."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    t = np.linspace(0, 390, 500)
    temp = sac305_profile(t)

    fig, ax = plt.subplots(figsize=(12, 6))

    # --- Zone shading ---
    # Preheat ramp (0–90s)
    ax.axvspan(0, 90, alpha=0.10, color='orange', label='Preheat Ramp')
    # Thermal soak (90–180s)
    ax.axvspan(90, 180, alpha=0.10, color='gold', label='Thermal Soak')
    # Ramp to reflow + reflow (180–270s)
    ax.axvspan(180, 270, alpha=0.12, color='red', label='Reflow')
    # Cooling (270–370s)
    ax.axvspan(270, 370, alpha=0.10, color='blue', label='Cooling')

    # --- Liquidus line at 217°C ---
    ax.axhline(y=217, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(375, 220, '217°C liquidus', fontsize=8, color='red', va='bottom')

    # --- Peak temperature line ---
    ax.axhline(y=249, color='darkred', linestyle=':', linewidth=1, alpha=0.5)
    ax.text(375, 251, '249°C peak', fontsize=8, color='darkred', va='bottom')

    # --- Profile curve ---
    ax.plot(t, temp, color='black', linewidth=2.5, zorder=5)

    # --- Zone labels ---
    ax.text(45, 30, 'PREHEAT\nRAMP\n~1.4°C/s', ha='center', va='bottom',
            fontsize=9, fontweight='bold', color='darkorange')
    ax.text(135, 30, 'THERMAL\nSOAK\n~0.28°C/s', ha='center', va='bottom',
            fontsize=9, fontweight='bold', color='goldenrod')
    ax.text(225, 30, 'REFLOW\nTAL = 60s', ha='center', va='bottom',
            fontsize=9, fontweight='bold', color='darkred')
    ax.text(320, 30, 'COOLING\n~1.9°C/s', ha='center', va='bottom',
            fontsize=9, fontweight='bold', color='darkblue')

    # --- Key point markers (from Chip Quik SMD291SNL datasheet) ---
    key_times =  [0,    90,    180,    210,    240,          270,    370]
    key_temps =  [25,  150,    175,    217,    249,          217,     25]
    key_labels = ['25°C', '150°C', '175°C', '217°C', '249°C\n(peak)', '217°C', '25°C']
    for tt, tp, lbl in zip(key_times, key_temps, key_labels):
        ax.plot(tt, tp, 'ko', markersize=5, zorder=6)
        ax.annotate(lbl, (tt, tp), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=7.5, color='black')

    # --- Axes ---
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_title('SAC305 No-Clean Solder Paste — Reflow Profile\n'
                 '(Chip Quik SMD291SNL, Sn96.5/Ag3.0/Cu0.5)',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(-5, 410)
    ax.set_ylim(0, 280)
    ax.grid(True, linestyle='--', alpha=0.3)

    # --- Legend ---
    handles = [
        mpatches.Patch(color='orange', alpha=0.3,
                       label='Preheat Ramp (0–90s)'),
        mpatches.Patch(color='gold', alpha=0.3,
                       label='Thermal Soak (90–180s)'),
        mpatches.Patch(color='red', alpha=0.3,
                       label='Reflow (180–270s, TAL=60s)'),
        mpatches.Patch(color='blue', alpha=0.3,
                       label='Cooling (270–370s)'),
    ]
    ax.legend(handles=handles, loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.show()


class ReflowMonitor(object):
    """Real-time reflow profile monitor.

    Shows two windows simultaneously:
      - OpenCV: thermal camera heatmap with HUD overlay
      - matplotlib: live temperature curves vs SAC305 reference profile

    Auto-starts the reflow timer when average temperature exceeds a threshold
    and holds above it for a configurable number of seconds (debounce).
    """

    def __init__(self, cameraFeed: CameraFeed, scale: int = 3,
                 start_threshold: float = 35.0,
                 start_hold_seconds: float = 5.0):
        self.cameraFeed: CameraFeed = cameraFeed
        self.thermalData: ThermalData = ThermalData(cameraFeed)
        self.scale: int = scale
        self.start_threshold: float = start_threshold
        self.start_hold_seconds: float = start_hold_seconds

        # Reflow state
        self.started: bool = False
        self.start_time: Optional[float] = None
        # Tracks when the temperature first crossed the threshold (for debounce)
        self._threshold_since: Optional[float] = None

        # Temperature history lists
        self.times: List[float] = []
        self.avg_temps: List[float] = []
        self.max_temps: List[float] = []
        self.center_temps: List[float] = []

        # OpenCV settings
        self.alpha: float = 1.0
        self.colormap_id: int = 0

        # Directories
        self._script_dir: str = os.path.dirname(os.path.abspath(__file__))
        self._plots_dir: str = os.path.join(self._script_dir, "plots")
        os.makedirs(self._plots_dir, exist_ok=True)

        # Audio feedback state
        self._sound_dir: str = os.path.join(self._script_dir, "sound")
        self._sound_too_slow: str = os.path.join(self._sound_dir, "too_slow.wav")
        self._sound_too_fast: str = os.path.join(self._sound_dir, "too_fast.wav")
        self._sound_good: str = os.path.join(self._sound_dir, "good.wav")
        self._last_sound_time: float = 0.0
        self._sound_cooldown: float = 5.0  # seconds between sounds
        self._last_pace: Optional[str] = None  # "slow", "fast", "good"
        self._pace_tolerance: float = 10.0  # °C deviation from target
        # Single-sound playback: only one afplay process at a time
        self._sound_process: Optional[subprocess.Popen] = None
        # Priority grace period: non-priority sounds are suppressed until this time
        self._sound_priority_until: float = 0.0
        self._sound_priority_grace: float = 3.0  # seconds to let phase sounds finish

        # Phase announcement sounds (played once at the start of each phase)
        self._sound_preheat: str = os.path.join(self._sound_dir, "preheat.wav")
        self._sound_soak: str = os.path.join(self._sound_dir, "soak.wav")
        self._sound_reflow: str = os.path.join(self._sound_dir, "reflow.wav")
        self._sound_cooling: str = os.path.join(self._sound_dir, "cooling.wav")
        self._current_phase: Optional[str] = None  # tracks current phase name

        # Rotation state: 0, 1, 2, or 3 (number of 90° clockwise rotations)
        self.rotation: int = 0

        # ROI (Region of Interest) selection state
        # roi stores (x1, y1, x2, y2) in sensor pixel coordinates (256x192),
        # None means use the full frame
        self.roi: Optional[Tuple[int, int, int, int]] = None
        self._roi_drawing: bool = False
        self._roi_start: Tuple[int, int] = (0, 0)  # mouse-down in window coords
        self._roi_end: Tuple[int, int] = (0, 0)     # current mouse pos in window coords

        # Matplotlib update throttling: only redraw the plot at this interval
        self._plot_update_interval: float = 1.0  # seconds
        self._last_plot_update: float = 0.0

        # Will be set during setup
        self.fig = None
        self.ax = None
        self.line_avg = None
        self.line_max = None
        self.line_center = None
        self.status_text = None
        self._bg = None  # cached background for blitting
        self.windowWidth: int = 0
        self.windowHeight: int = 0

    def setup_plot(self) -> None:
        """Create the matplotlib figure with SAC305 reference and empty live lines."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        plt.ion()  # interactive mode for real-time updates

        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        ax = self.ax

        # --- SAC305 reference profile (static) ---
        t_ref = np.linspace(0, 390, 500)
        temp_ref = sac305_profile(t_ref)
        ax.plot(t_ref, temp_ref, color='black', linewidth=2.5, zorder=3,
                label='SAC305 target profile')

        # --- Zone shading ---
        ax.axvspan(0, 90, alpha=0.08, color='orange')
        ax.axvspan(90, 180, alpha=0.08, color='gold')
        ax.axvspan(180, 270, alpha=0.10, color='red')
        ax.axvspan(270, 370, alpha=0.08, color='blue')

        # --- Liquidus and peak lines ---
        ax.axhline(y=217, color='red', linestyle='--', linewidth=1, alpha=0.6)
        ax.text(375, 220, '217°C liquidus', fontsize=7, color='red', va='bottom')
        ax.axhline(y=249, color='darkred', linestyle=':', linewidth=1, alpha=0.4)
        ax.text(375, 251, '249°C peak', fontsize=7, color='darkred', va='bottom')

        # --- Zone labels ---
        ax.text(45, 10, 'PREHEAT', ha='center', fontsize=8,
                fontweight='bold', color='darkorange', alpha=0.7)
        ax.text(135, 10, 'SOAK', ha='center', fontsize=8,
                fontweight='bold', color='goldenrod', alpha=0.7)
        ax.text(225, 10, 'REFLOW', ha='center', fontsize=8,
                fontweight='bold', color='darkred', alpha=0.7)
        ax.text(320, 10, 'COOLING', ha='center', fontsize=8,
                fontweight='bold', color='darkblue', alpha=0.7)

        # --- Live data lines (empty, will be updated in the loop) ---
        self.line_avg, = ax.plot([], [], color='dodgerblue', linewidth=2,
                                 zorder=5, label='Avg temp (live)')
        self.line_max, = ax.plot([], [], color='red', linewidth=1.5,
                                 linestyle='-', alpha=0.7, zorder=4,
                                 label='Max temp (live)')
        self.line_center, = ax.plot([], [], color='limegreen', linewidth=1.5,
                                    linestyle='-', alpha=0.7, zorder=4,
                                    label='Center pixel (live)')

        # --- Status text (upper right) ---
        self.status_text = ax.text(
            0.98, 0.97, 'Waiting for start...',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            horizontalalignment='right', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='gray', alpha=0.9),
            zorder=10
        )

        # --- Axes ---
        ax.set_xlabel('Time (seconds)', fontsize=11)
        ax.set_ylabel('Temperature (°C)', fontsize=11)
        ax.set_title('SAC305 Reflow Monitor — Live',
                     fontsize=13, fontweight='bold')
        ax.set_xlim(-5, 400)
        ax.set_ylim(0, 280)
        ax.grid(True, linestyle='--', alpha=0.3)

        # --- Legend ---
        handles = [
            mpatches.Patch(color='orange', alpha=0.25, label='Preheat (0-90s)'),
            mpatches.Patch(color='gold', alpha=0.25, label='Soak (90-180s)'),
            mpatches.Patch(color='red', alpha=0.25, label='Reflow (180-270s)'),
            mpatches.Patch(color='blue', alpha=0.25, label='Cooling (270-370s)'),
        ]
        ax.legend(handles=handles + [self.line_avg, self.line_max,
                  self.line_center, ax.lines[0]],
                  loc='upper left', fontsize=8)

        self.fig.tight_layout()
        self.fig.canvas.draw()
        # Cache the static background (zones, reference curve, axes) for blitting
        self._bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self.fig.canvas.flush_events()

    def setup_cv_window(self) -> None:
        """Create the OpenCV heatmap window."""
        sensor_width = self.cameraFeed.cameraParameters.frameWidth
        sensor_height = self.cameraFeed.cameraParameters.frameHeight // 2
        self.windowWidth = sensor_width * self.scale
        self.windowHeight = sensor_height * self.scale
        cv2.namedWindow('Thermal - Reflow Monitor', cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow('Thermal - Reflow Monitor',
                         self.windowWidth, self.windowHeight)

    def _window_to_sensor(self, wx: int, wy: int) -> Tuple[int, int]:
        """Convert window pixel coordinates to sensor pixel coordinates.

        Accounts for self.scale and self.rotation so the ROI rectangle drawn
        on the (possibly rotated) heatmap maps back to the correct region in
        the raw 256x192 thermal data grid.
        """
        sensor_w = self.cameraFeed.cameraParameters.frameWidth       # 256
        sensor_h = self.cameraFeed.cameraParameters.frameHeight // 2  # 192
        s = self.scale

        if self.rotation == 0:
            sx = wx // s
            sy = wy // s
        elif self.rotation == 1:  # 90° CW: displayed is (H*s, W*s)
            # In the rotated image, x spans sensor_h and y spans sensor_w
            sx = wy // s
            sy = (sensor_h - 1) - (wx // s)
        elif self.rotation == 2:  # 180°
            sx = (sensor_w - 1) - (wx // s)
            sy = (sensor_h - 1) - (wy // s)
        elif self.rotation == 3:  # 90° CCW
            sx = (sensor_w - 1) - (wy // s)
            sy = wx // s
        else:
            sx = wx // s
            sy = wy // s

        # Clamp to valid sensor range
        sx = max(0, min(sx, sensor_w - 1))
        sy = max(0, min(sy, sensor_h - 1))
        return sx, sy

    def _sensor_to_window(self, sx: int, sy: int) -> Tuple[int, int]:
        """Convert sensor pixel coordinates to window pixel coordinates.

        Inverse of _window_to_sensor. Used to draw the ROI rectangle on
        the heatmap overlay.
        """
        sensor_w = self.cameraFeed.cameraParameters.frameWidth       # 256
        sensor_h = self.cameraFeed.cameraParameters.frameHeight // 2  # 192
        s = self.scale

        if self.rotation == 0:
            wx = sx * s
            wy = sy * s
        elif self.rotation == 1:  # 90° CW
            wx = (sensor_h - 1 - sy) * s
            wy = sx * s
        elif self.rotation == 2:  # 180°
            wx = (sensor_w - 1 - sx) * s
            wy = (sensor_h - 1 - sy) * s
        elif self.rotation == 3:  # 90° CCW
            wx = sy * s
            wy = (sensor_w - 1 - sx) * s
        else:
            wx = sx * s
            wy = sy * s

        return wx, wy

    def _select_roi(self) -> None:
        """Interactive ROI selection phase (two-click model).

        Uses a trackpad-friendly two-click approach instead of click-and-drag:
          1. First click sets the first corner.
          2. Move the cursor (no button held) -- the rectangle live-previews.
          3. Second click locks the second corner.
          4. Press ENTER to confirm, or ESC to skip (full frame).

        This works reliably on macOS trackpads where click-and-drag events
        are often not reported continuously by OpenCV's HighGUI.
        """
        window_name = 'Thermal - Reflow Monitor'
        # _roi_clicks: 0 = no corner set, 1 = first corner set, 2 = both set
        roi_clicks = 0

        def _mouse_cb(event, x, y, flags, param):
            nonlocal roi_clicks
            if event == cv2.EVENT_LBUTTONDOWN:
                if roi_clicks == 0:
                    # First click: set first corner
                    self._roi_start = (x, y)
                    self._roi_end = (x, y)
                    roi_clicks = 1
                elif roi_clicks == 1:
                    # Second click: lock second corner
                    self._roi_end = (x, y)
                    roi_clicks = 2
                else:
                    # Already have both corners -- reset and start over
                    self._roi_start = (x, y)
                    self._roi_end = (x, y)
                    roi_clicks = 1
            elif event == cv2.EVENT_MOUSEMOVE:
                if roi_clicks == 1:
                    # Live preview: update second corner as cursor moves
                    self._roi_end = (x, y)

        cv2.setMouseCallback(window_name, _mouse_cb)

        logger.info("ROI selection mode. Click first corner, move to second "
                     "corner, click again. Then press ENTER to confirm. "
                     "Press ESC to skip (full frame).")

        while True:
            # Read a frame and build heatmap
            self.thermalData.extract_temperatures_fast()
            imdata = self.thermalData.imdata
            if imdata is None:
                continue

            bgr = cv2.cvtColor(imdata, cv2.COLOR_YUV2BGR_YUYV)
            bgr = cv2.convertScaleAbs(bgr, alpha=self.alpha)
            bgr = cv2.resize(bgr, (self.windowWidth, self.windowHeight),
                             interpolation=cv2.INTER_CUBIC)
            heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_JET)

            # Apply rotation
            if self.rotation == 1:
                heatmap = cv2.rotate(heatmap, cv2.ROTATE_90_CLOCKWISE)
            elif self.rotation == 2:
                heatmap = cv2.rotate(heatmap, cv2.ROTATE_180)
            elif self.rotation == 3:
                heatmap = cv2.rotate(heatmap, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Draw the rectangle (in-progress or finalized)
            x1, y1 = self._roi_start
            x2, y2 = self._roi_end
            if roi_clicks >= 1 and (x1, y1) != (x2, y2):
                # Green while adjusting, cyan once locked
                color = (0, 255, 0) if roi_clicks == 1 else (255, 255, 0)
                cv2.rectangle(heatmap, (x1, y1), (x2, y2), color, 2)

            # HUD instructions
            cv2.rectangle(heatmap, (0, 0), (380, 52), (0, 0, 0), -1)
            if roi_clicks == 0:
                hint = 'Click to set first corner'
            elif roi_clicks == 1:
                hint = 'Move cursor, click to set second corner'
            else:
                hint = 'ROI set! Press ENTER to confirm'
            cv2.putText(heatmap, hint, (10, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(heatmap,
                        'ENTER=confirm  ESC=skip  R=rotate',
                        (10, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (200, 200, 200), 1, cv2.LINE_AA)
            if roi_clicks >= 1:
                cv2.putText(heatmap,
                            'Click again to restart selection',
                            (10, 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            (150, 150, 150), 1, cv2.LINE_AA)

            cv2.imshow(window_name, heatmap)

            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # ENTER
                if roi_clicks >= 1 and (x1, y1) != (x2, y2):
                    # Convert the two corners to sensor coordinates
                    sx1, sy1 = self._window_to_sensor(x1, y1)
                    sx2, sy2 = self._window_to_sensor(x2, y2)
                    # Normalise so (x1,y1) is top-left and (x2,y2) is bottom-right
                    roi_x1 = min(sx1, sx2)
                    roi_y1 = min(sy1, sy2)
                    roi_x2 = max(sx1, sx2)
                    roi_y2 = max(sy1, sy2)
                    # Ensure at least a 2x2 region
                    roi_x2 = max(roi_x2, roi_x1 + 2)
                    roi_y2 = max(roi_y2, roi_y1 + 2)
                    self.roi = (roi_x1, roi_y1, roi_x2, roi_y2)
                    logger.info(f"ROI confirmed: sensor coords "
                                f"({roi_x1}, {roi_y1}) -> ({roi_x2}, {roi_y2})")
                else:
                    logger.info("No ROI drawn, using full frame.")
                    self.roi = None
                break
            elif key == 27:  # ESC
                logger.info("ROI selection skipped, using full frame.")
                self.roi = None
                break
            elif key == ord('r'):
                self.rotation = (self.rotation + 1) % 4
                logger.info(f"Rotation set to {self.rotation * 90}°")

        # Remove the mouse callback for the monitoring phase
        cv2.setMouseCallback(window_name, lambda *args: None)

    def _render_heatmap(self, avg_temp: float, max_temp: float,
                        center_temp: float,
                        pace: Optional[str] = None) -> None:
        """Convert imdata to heatmap and show with HUD overlay in OpenCV."""
        imdata = self.thermalData.imdata
        if imdata is None:
            return

        # YUYV to BGR
        bgr = cv2.cvtColor(imdata, cv2.COLOR_YUV2BGR_YUYV)
        bgr = cv2.convertScaleAbs(bgr, alpha=self.alpha)
        bgr = cv2.resize(bgr, (self.windowWidth, self.windowHeight),
                         interpolation=cv2.INTER_CUBIC)

        # Apply colormap
        heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_JET)

        # Draw ROI rectangle overlay (before rotation, in unrotated window coords)
        if self.roi is not None:
            rx1, ry1, rx2, ry2 = self.roi
            # Convert sensor coords to unrotated window coords (just scale)
            wx1, wy1 = rx1 * self.scale, ry1 * self.scale
            wx2, wy2 = rx2 * self.scale, ry2 * self.scale
            cv2.rectangle(heatmap, (wx1, wy1), (wx2, wy2),
                          (0, 255, 0), 2)

        # HUD overlay
        cv2.rectangle(heatmap, (0, 0), (220, 110), (0, 0, 0), -1)

        if self.started:
            elapsed = time.time() - self.start_time
            elapsed_str = time.strftime("%M:%S", time.gmtime(elapsed))
            cv2.putText(heatmap, f'REFLOW: {elapsed_str}', (10, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 255), 1,
                        cv2.LINE_AA)
        else:
            if self._threshold_since is not None:
                held = time.time() - self._threshold_since
                remaining = max(0, self.start_hold_seconds - held)
                cv2.putText(heatmap,
                            f'Starting in {remaining:.1f}s...', (10, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 200, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(heatmap, 'Waiting for start...', (10, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(heatmap,
                        f'Threshold: {self.start_threshold:.0f} C '
                        f'(hold {self.start_hold_seconds:.0f}s)',
                        (10, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1,
                        cv2.LINE_AA)

        y_offset = 36 if self.started else 54
        cv2.putText(heatmap, f'Avg:  {avg_temp:.1f} C', (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(heatmap, f'Max:  {max_temp:.1f} C', (10, y_offset + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 128, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(heatmap, f'Ctr:  {center_temp:.1f} C',
                    (10, y_offset + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 128), 1,
                    cv2.LINE_AA)

        # Pace indicator
        if pace is not None:
            pace_colors = {
                "slow": (0, 128, 255),    # orange
                "fast": (0, 0, 255),      # red
                "good": (0, 255, 0),      # green
            }
            pace_labels = {
                "slow": "TOO SLOW",
                "fast": "TOO FAST",
                "good": "GOOD",
            }
            color = pace_colors.get(pace, (200, 200, 200))
            label = pace_labels.get(pace, "")
            cv2.putText(heatmap, label, (10, y_offset + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
                        cv2.LINE_AA)

        # Apply rotation (0=none, 1=90° CW, 2=180°, 3=90° CCW)
        if self.rotation == 1:
            heatmap = cv2.rotate(heatmap, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation == 2:
            heatmap = cv2.rotate(heatmap, cv2.ROTATE_180)
        elif self.rotation == 3:
            heatmap = cv2.rotate(heatmap, cv2.ROTATE_90_COUNTERCLOCKWISE)

        cv2.imshow('Thermal - Reflow Monitor', heatmap)

    def _update_plot(self, avg_temp: float, max_temp: float,
                     center_temp: float,
                     pace: Optional[str] = None) -> None:
        """Update live matplotlib lines and status text.

        Throttled to self._plot_update_interval (default 1s) and uses
        blitting so only the dynamic artists (lines + text) are redrawn,
        keeping the static background (zones, reference curve, axes) cached.
        """
        now = time.time()

        # Always accumulate data, but only redraw at the throttled interval
        if self.started:
            elapsed = time.time() - self.start_time
            self.times.append(elapsed)
            self.avg_temps.append(avg_temp)
            self.max_temps.append(max_temp)
            self.center_temps.append(center_temp)

        if now - self._last_plot_update < self._plot_update_interval:
            # Still process Qt/Tk events so the window stays responsive
            self.fig.canvas.flush_events()
            return
        self._last_plot_update = now

        # --- Update status text ---
        if not self.started:
            if self._threshold_since is not None:
                held = time.time() - self._threshold_since
                remaining = max(0, self.start_hold_seconds - held)
                self.status_text.set_text(
                    f'Starting in {remaining:.1f}s...\n'
                    f'Avg: {avg_temp:.1f}°C  Max: {max_temp:.1f}°C\n'
                    f'Ctr: {center_temp:.1f}°C'
                )
            else:
                self.status_text.set_text(
                    f'Waiting for avg > {self.start_threshold:.0f}°C '
                    f'(hold {self.start_hold_seconds:.0f}s)\n'
                    f'Avg: {avg_temp:.1f}°C  Max: {max_temp:.1f}°C\n'
                    f'Ctr: {center_temp:.1f}°C'
                )
        else:
            elapsed = self.times[-1]

            self.line_avg.set_data(self.times, self.avg_temps)
            self.line_max.set_data(self.times, self.max_temps)
            self.line_center.set_data(self.times, self.center_temps)

            # Auto-extend x-axis if needed (requires full redraw to update ticks)
            if elapsed > self.ax.get_xlim()[1] - 20:
                self.ax.set_xlim(-5, elapsed + 60)
                # Axis limits changed -- must do a full redraw and re-cache bg
                self.fig.canvas.draw()
                self._bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)

            target = float(sac305_profile(np.array([elapsed]))[0])
            elapsed_str = time.strftime("%M:%S", time.gmtime(elapsed))
            pace_label = {"slow": "TOO SLOW", "fast": "TOO FAST",
                          "good": "GOOD"}.get(pace, "")
            self.status_text.set_text(
                f'Elapsed: {elapsed_str}  [{pace_label}]\n'
                f'Avg: {avg_temp:.1f}°C  (target: {target:.0f}°C)\n'
                f'Max: {max_temp:.1f}°C\n'
                f'Ctr: {center_temp:.1f}°C'
            )

        # --- Blit: restore cached background, redraw only dynamic artists ---
        if self._bg is not None:
            self.fig.canvas.restore_region(self._bg)
            self.ax.draw_artist(self.line_avg)
            self.ax.draw_artist(self.line_max)
            self.ax.draw_artist(self.line_center)
            self.ax.draw_artist(self.status_text)
            self.fig.canvas.blit(self.ax.bbox)
        else:
            self.fig.canvas.draw_idle()

        self.fig.canvas.flush_events()

    def _save_plot(self) -> None:
        """Save the current matplotlib figure to a timestamped PNG in the plots/ folder."""
        now = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(self._plots_dir, f"reflow_{now}.png")
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')
        logger.info(f"Plot saved to {filename}")

    def _play_sound(self, path: str, priority: bool = False) -> None:
        """Play a WAV file asynchronously using afplay (macOS).

        Only one sound plays at a time.  If a sound is already playing it is
        killed before the new one starts.

        Args:
            path: Path to the .wav file.
            priority: If True this is a high-priority sound (e.g. phase
                      announcement).  It will kill any running sound and set a
                      grace period during which non-priority sounds are
                      suppressed so the announcement can finish.
        """
        if not os.path.isfile(path):
            logger.warning(f"Sound file not found: {path}")
            return

        now = time.time()

        # If a priority sound is still in its grace period, skip non-priority
        if not priority and now < self._sound_priority_until:
            return

        # Kill any currently playing sound
        if self._sound_process is not None:
            if self._sound_process.poll() is None:  # still running
                self._sound_process.terminate()
            self._sound_process = None

        self._sound_process = subprocess.Popen(
            ["afplay", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # Priority sounds get a grace period so they aren't cut short
        if priority:
            self._sound_priority_until = now + self._sound_priority_grace

    def _check_pace(self, max_temp: float) -> Optional[str]:
        """Compare live max temp against the SAC305 target and play audio feedback.

        Returns the current pace: "slow", "fast", "good", or None if not started.
        Plays the corresponding sound if the pace changed or the cooldown expired.
        """
        if not self.started or self.start_time is None:
            return None

        elapsed = time.time() - self.start_time
        target = float(sac305_profile(np.array([elapsed]))[0])
        deviation = max_temp - target  # positive = ahead, negative = behind

        if deviation < -self._pace_tolerance:
            pace = "slow"
        elif deviation > self._pace_tolerance:
            pace = "fast"
        else:
            pace = "good"

        # Play sound if pace changed or cooldown expired
        now = time.time()
        pace_changed = (pace != self._last_pace)
        cooldown_ok = (now - self._last_sound_time >= self._sound_cooldown)

        if pace_changed or cooldown_ok:
            if pace == "slow":
                self._play_sound(self._sound_too_slow)
            elif pace == "fast":
                self._play_sound(self._sound_too_fast)
            else:
                self._play_sound(self._sound_good)
            self._last_sound_time = now
            self._last_pace = pace

        return pace

    def _get_phase(self, elapsed: float) -> str:
        """Return the current reflow phase name based on elapsed seconds.

        Phase boundaries (from SAC305 profile):
          Preheat:  0 – 90s
          Soak:    90 – 180s
          Reflow: 180 – 270s
          Cooling: 270s+
        """
        if elapsed < 90:
            return "preheat"
        elif elapsed < 180:
            return "soak"
        elif elapsed < 270:
            return "reflow"
        else:
            return "cooling"

    def _check_phase_transition(self) -> None:
        """Detect phase transitions and play the announcement sound once."""
        if not self.started or self.start_time is None:
            return

        elapsed = time.time() - self.start_time
        phase = self._get_phase(elapsed)

        if phase != self._current_phase:
            logger.info(f"Phase transition: {self._current_phase} -> {phase}")
            self._current_phase = phase
            sound_map = {
                "preheat": self._sound_preheat,
                "soak": self._sound_soak,
                "reflow": self._sound_reflow,
                "cooling": self._sound_cooling,
            }
            sound = sound_map.get(phase)
            if sound:
                self._play_sound(sound, priority=True)

    def run(self) -> None:
        """Main monitoring loop."""
        import matplotlib.pyplot as plt

        self.setup_plot()
        # Read one frame to initialize dimensions before creating cv window
        self.thermalData.extract_temperatures_fast()
        self.setup_cv_window()

        # --- ROI selection phase ---
        self._select_roi()

        logger.info("Reflow monitor started. Press Esc in the thermal window "
                     "to quit.")
        if self.roi is not None:
            logger.info(f"Using ROI: {self.roi}")
        else:
            logger.info("Using full frame (no ROI).")
        logger.info(f"Auto-start threshold: {self.start_threshold}°C "
                     f"(hold for {self.start_hold_seconds}s)")

        try:
            while True:
                avg_temp, max_temp, center_temp = \
                    self.thermalData.extract_temperatures_fast(roi=self.roi)

                # Auto-start detection with debounce
                if not self.started:
                    if avg_temp > self.start_threshold:
                        now = time.time()
                        if self._threshold_since is None:
                            self._threshold_since = now
                            logger.info(
                                f"Threshold crossed ({avg_temp:.1f}°C "
                                f"> {self.start_threshold}°C), "
                                f"holding for {self.start_hold_seconds}s...")
                        held_for = now - self._threshold_since
                        if held_for >= self.start_hold_seconds:
                            self.started = True
                            self.start_time = time.time()
                            logger.info(
                                f"Reflow started! Avg temp held above "
                                f"{self.start_threshold}°C for "
                                f"{self.start_hold_seconds}s")
                    else:
                        # Dropped below threshold, reset debounce
                        if self._threshold_since is not None:
                            logger.info(
                                f"Avg temp dropped to {avg_temp:.1f}°C, "
                                f"resetting hold timer.")
                        self._threshold_since = None

                # Check pace and play audio feedback
                pace = self._check_pace(max_temp)

                # Check for phase transitions and play announcement sounds
                self._check_phase_transition()

                # Update both windows
                self._render_heatmap(avg_temp, max_temp, center_temp, pace)
                self._update_plot(avg_temp, max_temp, center_temp, pace)

                # Handle key events (Esc to quit, 's' to save)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # Esc
                    logger.info("Esc pressed, stopping.")
                    break
                elif key == ord('s'):
                    self._save_plot()
                elif key == ord('r'):
                    self.rotation = (self.rotation + 1) % 4
                    logger.info(f"Rotation set to {self.rotation * 90}°")

        except KeyboardInterrupt:
            logger.info("Ctrl+C received, stopping.")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Save final plot, terminate ffmpeg, close cv2 and matplotlib."""
        import matplotlib.pyplot as plt

        # Save the final plot if we have any data
        if self.fig is not None and len(self.times) > 0:
            self._save_plot()

        # Kill any sound still playing
        if self._sound_process is not None and self._sound_process.poll() is None:
            self._sound_process.terminate()
            self._sound_process = None

        self.cameraFeed.close()
        cv2.destroyAllWindows()
        plt.close('all')
        logger.info("Reflow monitor stopped. Cleanup complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0,
                        help="avfoundation device index, run: ffmpeg -f avfoundation -list_devices true -i \"\"")
    parser.add_argument("--list-devices", action="store_true",
                        help="list avfoundation devices")
    parser.add_argument("--calibrate", action="store_true",
                        help="calibrate the thermal camera")
    parser.add_argument("--camera", choices=["ts001", "tc001"], default="ts001",
                        help="camera type, must be one of: ts001, tc001")
    parser.add_argument("--plot", action="store_true",
                        help="plot the sac305 profile")
    parser.add_argument("--reflow", action="store_true",
                        help="start real-time reflow monitoring mode")
    parser.add_argument("--start-threshold", type=float, default=35.0,
                        help="avg temperature threshold (°C) to auto-start "
                             "the reflow timer (default: 35.0)")
    parser.add_argument("--start-hold", type=float, default=5.0,
                        help="seconds the avg temp must stay above threshold "
                             "before the timer starts (default: 5.0)")
    args = parser.parse_args()

    def list_devices():
        subprocess.run(["ffmpeg", "-f", "avfoundation",
                        "-list_devices", "true", "-i", ""])

    if args.plot:
        plot_sac305_profile()
        sys.exit(0)

    if args.list_devices:
        list_devices()
        sys.exit(0)

    dev = args.device

    if args.reflow:
        logger.info("Starting reflow monitoring mode")

        camera_parameter: CameraParameters = None
        if args.camera == "tc001":
            camera_parameter = CameraParameters(256, 384, 25, "yuyv422")
        elif args.camera == "ts001":
            camera_parameter = CameraParameters(256, 384, 25, "yuyv422")

        camera_feed = CameraFeed(camera_parameter)
        camera_feed.open(dev)

        monitor = ReflowMonitor(
            cameraFeed=camera_feed,
            scale=3,
            start_threshold=args.start_threshold,
            start_hold_seconds=args.start_hold
        )
        monitor.run()
        sys.exit(0)

    if args.calibrate:
        logger.info("Calibrate the camera")

        camera_parameter: CameraParameters = None
        if args.camera == "tc001":
            camera_parameter = CameraParameters(
                256, 384, 25, "yuyv422"
            )
        elif args.camera == "ts001":
            camera_parameter = CameraParameters(
                256, 384, 25, "yuyv422"
            )

        camera_feed = CameraFeed(camera_parameter)
        thermal_data = ThermalData(camera_feed)
        thermal_view = ThermalView(thermal_data)

        camera_feed.open(args.device)
        thermal_data.extract_temperature_data()
        thermal_view.init_window()

        try:
            while True:
                thermal_data.extract_temperature_data()
                thermal_view.show()
        except:
            thermal_view.close()
            camera_feed.close()
            sys.exit(0)
