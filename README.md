# WatercolorVideoMaker
So, I wasted too many hours in Python because Adobe Photoshop was sh*tting itself again...

I needed to make a "watercolour style" stage backdrop video for an event in Zhejiang University, and I watched [this tutorial](https://www.youtube.com/watch?v=DV9Qz0qy4qE). All went well until it came time for me to EXPORT the video. I clicked the button in Photoshop to export a video, and... it just spun endlessly. After Effects was not in a cooperative mood either...

After waiting long enough to rethink my life choices, I gave up and vibe coded this script to reproduce the same layer stack in OpenCV, frame by frame, like a normal person in 2026 apparently has to do. I wasted all this time instead of studying for Discrete Math, and I still have many things I need to do in Discrete Math class...

## What this does
This script applies a watercolor-ish pipeline to every frame of a video:

1. Dry Brush (stylization)
2. Cutout (posterized region smoothing)
3. Smart Blur (bilateral)
4. Find Edges (multiplied in)
5. Paper texture multiply
6. Paper texture overlay at partial opacity

In other words: it automates the Photoshop tutorial effect, except it actually finishes.

## Why this exists
- Photoshop export stalled forever.
- After Effects was not in a cooperative mood either.
- I only needed a 3-minute background visual, not a spiritual journey through codec hell.
- Python/OpenCV lets me batch process reliably and move on with life.

## Requirements
- Python 3.9+
- OpenCV
- NumPy

Install dependencies:

```bash
pip install opencv-python numpy
```

## Usage

### Original version
```bash
python3 main.py \
  --input "/path/to/input.mov" \
  --paper "/path/to/paper_texture.jpg" \
  --output "/path/to/output.mp4"
```

### Fast version (recommended)
```bash
python3 main_fast.py \
  --input "/path/to/input.mov" \
  --paper "/path/to/paper_texture.jpg" \
  --output "/path/to/output.mp4" \
  --workers 4 \
  --scale 0.6 \
  --fast-mode
```

## Parameters (fast version)
- `--workers`  
  Number of parallel worker processes.  
  Start with number of performance cores (e.g. 4 on many Apple Silicon setups).

- `--scale`  
  Internal processing scale (`0.25` to `1.0`).  
  Lower = faster, softer image.  
  `1.0` = full quality, slowest.

- `--fast-mode`  
  Uses faster approximations for some effects.  
  Looks close enough for stage background visuals unless you're grading for Cannes.

## Tuning guide (because of course you need one)
- Faster preview: `--scale 0.5 --fast-mode`
- Balanced: `--scale 0.6 --fast-mode --workers 4`
- Better quality: `--scale 0.85` (optionally without `--fast-mode`)
- Max quality: `--scale 1.0` (bring patience and maybe tea)

## Audio note
OpenCV output is video-only in this pipeline.  
To reattach original audio with ffmpeg:

```bash
ffmpeg -y -i "/path/to/output.mp4" -i "/path/to/input.mov" \
-map 0:v:0 -map 1:a:0 -c:v copy -shortest "/path/to/output_with_audio.mp4"
```

## Known limitations
- Effect match is approximate and kinda sucks TBH (not Adobe’s proprietary internals).
- Heavy filters are still computationally expensive.
- If you push quality settings too high, render time will remind you that physics is real.

## Credits
- Visual style inspiration: the YouTube tutorial linked above.
- Emotional damage: Adobe export pipeline.
- Recovery plan: Python + OpenCV + stubbornness.

## Was I satisfied with the result?
Not really. I think the comic filters in Final Cut Pro were a lot better in the end and exported a lot faster. I should have just used them from the start.
