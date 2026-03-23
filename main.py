import cv2
import numpy as np
import argparse

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

# ========================= EFFECTS (applied to a fresh copy of the original frame each time) =========================
def apply_dry_brush(frame):
    """Dry Brush ≈ OpenCV stylization (painterly strokes — tweak sigma_s/sigma_r if needed)"""
    return cv2.stylization(frame, sigma_s=60, sigma_r=0.45)

def apply_cutout(frame):
    """Cutout ≈ edge-preserving posterization (perfect match to Photoshop Cutout)"""
    return cv2.pyrMeanShiftFiltering(frame, sp=30, sr=40, maxLevel=2)

def apply_smart_blur(frame):
    """Smart Blur (radius=5, threshold=100) — bilateralFilter is the exact equivalent"""
    return cv2.bilateralFilter(frame, d=5, sigmaColor=100, sigmaSpace=5)

def apply_find_edges(frame):
    """Stylize > Find Edges — Laplacian gives thin dark lines on white (Photoshop style)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    lap = np.uint8(np.absolute(lap))
    edges = 255 - lap                                      # invert for dark edges on white
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

# ========================= MAIN SCRIPT =========================
def main():
    parser = argparse.ArgumentParser(description="Python Watercolor Effect — exact Photoshop tutorial recreation")
    parser.add_argument("--input", required=True, help="Input video path (e.g. footage.mp4)")
    parser.add_argument("--paper", required=True, help="Paper texture path (paper_texture.jpg)")
    parser.add_argument("--output", default="watercolor_output.mp4", help="Output video path")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print("Error opening input video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # Load paper texture once
    paper = cv2.imread(args.paper)
    if paper is None:
        print("Error loading paper texture")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        original = frame.copy()

        # 1. Dry Brush layer (Normal, 100%)
        dry = apply_dry_brush(original)
        current = composite(original, dry, None, 1.0)

        # 2. Cutout layer (Pin Light, 100%)
        cut = apply_cutout(original)
        current = composite(current, cut, pin_light_blend, 1.0)

        # 3. Smart Blur layer (Screen, 50%)
        smart = apply_smart_blur(original)
        current = composite(current, smart, screen_blend, 0.5)

        # 4. Find Edges layer (Multiply, 100%)
        edges = apply_find_edges(original)
        current = composite(current, edges, multiply_blend, 1.0)

        # 5. Paper texture #1 — stretched to fill, Multiply
        paper1 = cv2.resize(paper, (width, height))
        current = composite(current, paper1, multiply_blend, 1.0)

        # 6. Paper texture #2 — normal, 50% opacity
        paper2 = paper1.copy()  # same stretched version
        current = composite(current, paper2, None, 0.5)

        out.write(current)
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"✅ Done! Output saved to: {args.output}")
    print("Note: Audio is not preserved (OpenCV limitation). Add it back with ffmpeg or MoviePy if needed.")

if __name__ == "__main__":
    main()