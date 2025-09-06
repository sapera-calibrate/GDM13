#!/usr/bin/env python3
"""
Compute difference metrics between 1.png and 2.png and save visual diff.

Outputs JSON:
{
  "mse": …,
  "psnr": …,
  "percent_identical": …,
  "percent_difference": …,
  "diff_image": "diff_visual.png"
}
If any step fails it emits {"error": "...", "trace": "..."} and exits ≠0
"""
import json, math, os, sys, traceback, argparse
try:
    import numpy as np
    from PIL import Image, ImageChops
except Exception as e:      # dependency missing
    traceback.print_exc()
    sys.exit(1)

IMAGES_DIR = "images"

def find_default_images():
    """Return first two .png images inside IMAGES_DIR or raise ValueError"""
    files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith('.png')]
    if len(files) < 2:
        raise ValueError(f"Need at least 2 .png images inside {IMAGES_DIR} to compare, found {len(files)}")
    files.sort()
    return os.path.join(IMAGES_DIR, files[0]), os.path.join(IMAGES_DIR, files[1])

def main() -> None:
    try:
        parser = argparse.ArgumentParser(description="Compute difference metrics between two images")
        parser.add_argument('image1', nargs='?', help='Path to first image')
        parser.add_argument('image2', nargs='?', help='Path to second image')
        args = parser.parse_args()

        if args.image1 and args.image2:
            img1_path, img2_path = args.image1, args.image2
        else:
            img1_path, img2_path = find_default_images()
            print(f"[INFO] Auto-selected images: {os.path.basename(img1_path)} & {os.path.basename(img2_path)}", file=sys.stderr)
        if not os.path.exists(img1_path) or not os.path.exists(img2_path):
            raise FileNotFoundError("1.png or 2.png not found in project root")

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        if img1.size != img2.size:
            raise ValueError(f"Dimension mismatch: {img1.size} vs {img2.size}")

        arr1 = np.asarray(img1, dtype=np.float32)
        arr2 = np.asarray(img2, dtype=np.float32)

        mse  = float(np.mean((arr1 - arr2) ** 2))
        psnr = float('inf') if mse == 0 else 20 * math.log10(255.0 / math.sqrt(mse))
        identical = int(np.sum(np.all(arr1 == arr2, axis=2)))
        percent_identical = identical * 100.0 / (arr1.shape[0] * arr1.shape[1])

        # visual diff
        diff_img = ImageChops.difference(img1, img2)
        enhanced = diff_img.point(lambda x: min(255, x * 10))
        diff_dir = os.path.dirname(img1_path) or '.'
        diff_path = os.path.join(diff_dir, 'diff_visual.png')
        enhanced.save(diff_path)

        print(json.dumps({
            'mse': mse,
            'psnr': psnr,
            'percent_identical': percent_identical,
            'percent_difference': 100.0 - percent_identical,
            'diff_image': diff_path
        }, indent=2))
    except Exception as exc:
        print(json.dumps({
            'error': str(exc),
            'trace': traceback.format_exc()
        }, indent=2))
        sys.exit(1)

if __name__ == '__main__':
    main()