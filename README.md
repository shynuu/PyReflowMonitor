# Reflow Monitor — Real-Time SAC305 Solder Profile Tracker

**Based on the work done in https://github.com/leswright1977/PyThermalCamera with modifications for MACOS support and features required for the reflow monitor. Requires a Topdon TC001/TS001 Thermal Camera or similar
Mix of own code + claude opus 4.6 model code**

A real-time reflow soldering monitor that uses a Topdon TC001 / TS001 thermal
camera to track board temperatures against the
[SAC305 (Chip Quik SMD291SNL)](https://www.chipquik.com/datasheets/SMD291SNL.pdf)
reflow profile.

## Features

- **Dual-window display** — OpenCV thermal heatmap + matplotlib live plot
  running side by side on the main thread (macOS compatible).
- **Smooth SAC305 reference curve** — piecewise exponential, cubic, parabolic
  and decay functions matching the datasheet profile shape (not linear
  interpolation).
- **ROI selection** — before monitoring starts, interactively select a Region
  of Interest by clicking two corners on the heatmap so only your PCB area is
  tracked.
- **Auto-start with debounce** — the reflow timer starts automatically when
  the average temperature in the ROI stays above a configurable threshold for a
  configurable hold period.
- **Three live temperature lines** — Average (blue), Max (red), Center pixel
  (green) plotted against the SAC305 reference curve.
- **Phase announcements** — audio cues (`preheat.wav`, `soak.wav`,
  `reflow.wav`, `cooling.wav`) play once at each phase transition.
- **Pace feedback** — audio alerts (`too_slow.wav`, `too_fast.wav`, `good.wav`)
  when the live max temp deviates from the target. Phase sounds take priority
  and suppress pace sounds briefly so announcements aren't cut short.
- **Rotation** — press `r` to rotate the thermal heatmap by 90 degrees
  (cycles through 0° / 90° / 180° / 270°).
- **Plot snapshots** — press `s` at any time to save the current plot, or the
  final plot is saved automatically on exit. All plots are saved to
  `src/plots/`.

## Requirements

- **macOS** (uses `ffmpeg` + `avfoundation` for camera capture and `afplay` for
  audio)
- Python 3.8+
- `numpy`
- `opencv-python` (`cv2`)
- `matplotlib`
- `ffmpeg` installed and on `PATH`

Install Python dependencies:

```bash
pip install numpy opencv-python matplotlib
```

## Usage

### List available cameras

```bash
python src/reflow.py --list-devices
```

This runs `ffmpeg -f avfoundation -list_devices true` to show available video
inputs. Note the device index for your thermal camera.

### Start the reflow monitor

```bash
python src/reflow.py --reflow --device <N>
```

Full options:

| Flag | Default | Description |
|------|---------|-------------|
| `--device <N>` | `0` | avfoundation device index |
| `--camera {ts001,tc001}` | `ts001` | Camera model |
| `--reflow` | — | Start reflow monitoring mode |
| `--start-threshold <°C>` | `35.0` | Avg temp threshold to auto-start the timer |
| `--start-hold <s>` | `5.0` | Seconds avg must stay above threshold before starting |
| `--plot` | — | Show the static SAC305 profile chart and exit |
| `--calibrate` | — | Run basic thermal view (no reflow features) |
| `--list-devices` | — | List avfoundation devices and exit |

Example with a custom threshold:

```bash
python src/reflow.py --reflow --device 1 --start-threshold 40 --start-hold 3
```

### Workflow

1. **ROI Selection** — the thermal heatmap opens first. Position your PCB in
   frame, then:
   - **Click** to set the first corner of the region of interest.
   - **Move** the cursor (no dragging needed — trackpad friendly).
   - **Click** again to lock the second corner.
   - Press **ENTER** to confirm, or **ESC** to skip and use the full frame.
   - Press **r** to rotate if needed.

2. **Monitoring** — the main loop starts. Two windows are shown:
   - **Thermal heatmap** (OpenCV) — live camera feed with a green ROI
     rectangle, HUD showing current temperatures, elapsed time, and pace
     indicator.
   - **Live plot** (matplotlib) — avg / max / center temperature curves drawn
     over the SAC305 reference profile with zone shading.

3. **Auto-start** — once the average temperature in the ROI exceeds the
   threshold and holds for the configured duration, the timer and live plotting
   begin automatically.

4. **Phase sounds** — at each phase boundary (preheat → soak → reflow →
   cooling) an announcement sound plays. Between announcements, pace feedback
   sounds indicate whether you're ahead, behind, or on target.

5. **Exit** — press **ESC** in the thermal window or **Ctrl+C** in the
   terminal. The final plot is saved automatically to `src/plots/`.

### Key bindings (during monitoring)

| Key | Action |
|-----|--------|
| `ESC` | Quit and save final plot |
| `s` | Save current plot snapshot |
| `r` | Rotate heatmap by 90° |

## SAC305 Reflow Profile

The reference curve uses smooth piecewise functions matching the
[Chip Quik SMD291SNL datasheet](https://www.chipquik.com/datasheets/SMD291SNL.pdf):

| Phase | Time | Temp | Curve shape |
|-------|------|------|-------------|
| Preheat | 0 – 90 s | 25 → 150 °C | Exponential approach |
| Soak + Ramp | 90 – 210 s | 150 → 175 → 217 °C | Smooth cubic S-curve |
| Reflow (TAL) | 210 – 270 s | 217 → 249 → 217 °C | Parabolic peak |
| Cooling | 270 – 370 s | 217 → 25 °C | Exponential decay |

## Project Structure

```
src/
├── reflow.py              # Main script (all classes and CLI)
├── REFLOW_README.md       # This file
├── plots/                 # Saved plot snapshots (auto-created)
│   └── reflow_YYYYMMDD-HHMMSS.png
└── sound/                 # Audio feedback files
    ├── preheat.wav        # Phase: preheat started
    ├── soak.wav           # Phase: soak started
    ├── reflow.wav         # Phase: reflow started
    ├── cooling.wav        # Phase: cooling started
    ├── too_slow.wav       # Pace: behind target
    ├── too_fast.wav       # Pace: ahead of target
    └── good.wav           # Pace: on target
```

## Architecture

```
CameraFeed (ffmpeg pipe)
    └──▶ ThermalData.extract_temperatures_fast(roi=...)
              │
              ├──▶ avg / max / center temps
              │         │
              │         ├──▶ Auto-start detection (threshold + debounce)
              │         ├──▶ _check_pace()     → pace feedback sound
              │         ├──▶ _check_phase_transition() → phase announcement
              │         ├──▶ _render_heatmap()  → OpenCV window
              │         └──▶ _update_plot()     → matplotlib window (throttled + blitting)
              │
              └──▶ imdata (YUYV frame for heatmap rendering)
```

Key design decisions:

- **Single-threaded** — both OpenCV and matplotlib require the main thread on
  macOS. `plt.ion()` + `canvas.flush_events()` and `cv2.waitKey(1)` alternate
  in the same loop.
- **Vectorized temps** — `extract_temperatures_fast()` computes all
  temperatures in one numpy operation (no Python loops over 49k pixels).
- **Throttled plot updates** — matplotlib redraws at most once per second using
  blitting (only the dynamic lines and text are redrawn; the static background
  is cached).
- **Single sound playback** — only one `afplay` process runs at a time. Phase
  sounds kill any playing sound and get a grace period during which pace sounds
  are suppressed.
- **Two-click ROI** — avoids click-and-drag which is unreliable on macOS
  trackpads.
