import cv2
import numpy as np
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

# ========================= BLENDING MODES (exact Photoshop matches) =========================
def multiply_blend(base, overlay):
    """Photoshop Multiply"""
    return np.clip((base.astype(np.float32) * overlay.astype(np.float32) / 255.0), 0, 255).astype(np.uint8)

def screen_blend(base, overlay):
    """Photoshop Screen"""
    return np.clip(255 - ((255 - base.astype(np.float32)) * (255 - overlay.astype(np.float32)) / 255.0), 0, 255).astype(np.uint8)

def pin_light_blend(base, overlay):
    """Photoshop Pin Light (exact formula)"""
    base_f = base.astype(np.float32)
    overlay_f = overlay.astype(np.float32)
    result = np.where(overlay_f < 128,
                      np.minimum(base_f, 2 * overlay_f),
                      np.maximum(base_f, 2 * overlay_f - 255))
    return np.clip(result, 0, 255).astype(np.uint8)

def composite(base, overlay, blend_func=None, opacity=1.0):
    """Applies blend mode then opacity (exactly how Photoshop layers work)"""
    if blend_func is None:  # Normal blend
        blended = overlay
    else:
        blended = blend_func(base, overlay)
    return cv2.addWeighted(base, 1 - opacity, blended, opacity, 0)

# ========================= EFFECTS =========================
def apply_dry_brush(frame, fast_mode=False):
    """
    Dry Brush:
    - normal mode: cv2.stylization (higher quality, slower)
    - fast mode: edge-preserving + mild sharpen (faster)
    """
    if not fast_mode:
        return cv2.stylization(frame, sigma_s=100, sigma_r=0.6)

    # Fast approximation
    sm = cv2.edgePreservingFilter(frame, flags=1, sigma_s=40, sigma_r=0.35)
    blur = cv2.GaussianBlur(sm, (0, 0), 1.0)
    sharp = cv2.addWeighted(sm, 1.25, blur, -0.25, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)

def apply_cutout(frame, fast_mode=False):
    """
    Cutout:
    - normal mode: mean-shift (slow)
    - fast mode: color quantization + median blur
    """
    if not fast_mode:
        return cv2.pyrMeanShiftFiltering(frame, sp=30, sr=40, maxLevel=2)

    # Fast posterize-like approximation
    levels = 8
    q = (frame // (256 // levels)) * (256 // levels)
    return cv2.medianBlur(q, 5)

def apply_smart_blur(frame, fast_mode=False):
    """
    Smart Blur:
    - normal mode: bilateralFilter
    - fast mode: lighter bilateral
    """
    if not fast_mode:
        return cv2.bilateralFilter(frame, d=9, sigmaColor=150, sigmaSpace=9)
    return cv2.bilateralFilter(frame, d=3, sigmaColor=45, sigmaSpace=3)

def apply_find_edges(frame):
    """Stylize > Find Edges approximation"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    lap = np.uint8(np.absolute(lap))
    edges = 255 - lap
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

# ========================= WORKER =========================
# Globals initialized per process
G_PAPER = None
G_WIDTH = None
G_HEIGHT = None
G_SCALE = 1.0
G_FAST = False

def _init_worker(paper_path, width, height, scale, fast_mode):
    global G_PAPER, G_WIDTH, G_HEIGHT, G_SCALE, G_FAST

    # OpenCV runtime tuning per worker
    cv2.setUseOptimized(True)
    cv2.setNumThreads(0)

    G_WIDTH = width
    G_HEIGHT = height
    G_SCALE = scale
    G_FAST = fast_mode

    paper = cv2.imread(paper_path)
    if paper is None:
        raise RuntimeError(f"Could not load paper texture in worker: {paper_path}")
    G_PAPER = cv2.resize(paper, (width, height), interpolation=cv2.INTER_AREA)

def _process_one(args):
    idx, frame = args

    original_full = frame

    # Optional working downscale
    if G_SCALE < 1.0:
        sw = max(1, int(G_WIDTH * G_SCALE))
        sh = max(1, int(G_HEIGHT * G_SCALE))
        original = cv2.resize(original_full, (sw, sh), interpolation=cv2.INTER_AREA)
        paper_work = cv2.resize(G_PAPER, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        original = original_full
        paper_work = G_PAPER

    # 1. Dry Brush layer (Normal, 100%)
    dry = apply_dry_brush(original, fast_mode=G_FAST)
    current = composite(original, dry, None, 1.0)

    # 2. Cutout layer (Pin Light, 100%)
    cut = apply_cutout(original, fast_mode=G_FAST)
    current = composite(current, cut, pin_light_blend, 1.0)

    # 3. Smart Blur layer (Screen, 50%)
    smart = apply_smart_blur(original, fast_mode=G_FAST)
    current = composite(current, smart, screen_blend, 0.5)

    # 4. Find Edges layer (Multiply, 100%)
    edges = apply_find_edges(original)
    current = composite(current, edges, multiply_blend, 1.0)

    # 5. Paper texture #1 — Multiply
    current = composite(current, paper_work, multiply_blend, 1.0)

    # 6. Paper texture #2 — Normal, 50%
    current = composite(current, paper_work, None, 0.3)

    # Upscale back to output size if needed
    if G_SCALE < 1.0:
        current = cv2.resize(current, (G_WIDTH, G_HEIGHT), interpolation=cv2.INTER_LINEAR)

    return idx, current

# ========================= MAIN SCRIPT =========================
def main():
    parser = argparse.ArgumentParser(
        description="Fast Watercolor Effect Video Processor (parallel + quality/speed controls)"
    )
    parser.add_argument("--input", required=True, help="Input video path (e.g. footage.mp4)")
    parser.add_argument("--paper", required=True, help="Paper texture path (paper_texture.jpg)")
    parser.add_argument("--output", default="watercolor_output.mp4", help="Output video path")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help="Number of worker processes (default: CPU cores - 1)")
    parser.add_argument("--buffer", type=int, default=64,
                        help="Max in-flight frames waiting for results")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Internal processing scale (0.25..1.0). Lower = faster.")
    parser.add_argument("--fast-mode", action="store_true",
                        help="Use faster approximations for dry brush/cutout/smart blur")
    args = parser.parse_args()

    args.scale = max(0.25, min(1.0, args.scale))
    args.workers = max(1, args.workers)
    args.buffer = max(4, args.buffer)

    # Main-process OpenCV tuning
    cv2.setUseOptimized(True)
    cv2.setNumThreads(0)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print("Error opening input video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    if not out.isOpened():
        print("Error opening output video writer")
        cap.release()
        return

    print(f"Starting...")
    print(f"  Input: {args.input}")
    print(f"  Paper: {args.paper}")
    print(f"  Output: {args.output}")
    print(f"  Size: {width}x{height} @ {fps:.2f} fps")
    print(f"  Workers: {args.workers}")
    print(f"  Buffer: {args.buffer}")
    print(f"  Scale: {args.scale}")
    print(f"  Fast mode: {args.fast_mode}")

    next_to_write = 0
    frame_idx = 0
    pending = {}
    end_of_stream = False
    processed_count = 0

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(args.paper, width, height, args.scale, args.fast_mode),
    ) as pool:

        in_flight = {}

        while True:
            # Feed workers while buffer has room
            while (not end_of_stream) and (len(in_flight) < args.buffer):
                ret, frame = cap.read()
                if not ret:
                    end_of_stream = True
                    break
                fut = pool.submit(_process_one, (frame_idx, frame))
                in_flight[fut] = frame_idx
                frame_idx += 1

            if not in_flight:
                break  # done

            # Pull completed futures (non-blocking-ish pass)
            done_now = [f for f in list(in_flight.keys()) if f.done()]
            if not done_now:
                # If nothing done yet, wait for one result to avoid spin
                f = next(iter(in_flight.keys()))
                idx, result = f.result()
                pending[idx] = result
                del in_flight[f]
            else:
                for f in done_now:
                    idx, result = f.result()
                    pending[idx] = result
                    del in_flight[f]

            # Write in strict order
            while next_to_write in pending:
                out.write(pending.pop(next_to_write))
                next_to_write += 1
                processed_count += 1
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} frames...")

    cap.release()
    out.release()

    print(f"✅ Done! Output saved to: {args.output}")
    print("Note: Audio is not preserved (OpenCV limitation).")
    print("Tip: Add original audio back with ffmpeg:")
    print(f'ffmpeg -y -i "{args.output}" -i "{args.input}" -c:v copy -map 0:v:0 -map 1:a:0 -shortest "with_audio.mp4"')

if __name__ == "__main__":
    main()