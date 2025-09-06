<p align="center">
  <img src="docs/protrace_logo.png" alt="ProTrace logo" width="180"/>
</p>

# ProTrace GDM13 — Grid-Diffusion Mapping (13 refinements)  
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)  ![Status](https://img.shields.io/badge/status-active-success)


> **GDM** = **G**rid + **D**ithering + importance **M**apping  
> **13** = refinement passes / hierarchical depth


A robust image fingerprinting system that generates compact fingerprints using keyed chaotic coordinates, block SPN (Substitution-Permutation Network) hashes, and checksums. Designed for verification of images without transferring full image data.

## Why GDM13 is different — not just _another pHash_

- **Not another pHash** – beyond perceptual similarity; bit-level tamper granularity → proportional output flips.
- **Grid + Dithering + Mapping** – hybrid sampler mixes uniform coverage, error-diffusion and ISCC-style saliency to select the most informative pixels.
- **Crypto-native design** – keyed SPN hashes, Bloom/Verkle style proofs; bridges cryptography & vision.
- **Protocol-level thinking** – designed for large-scale, verifiable deployment (marketplaces, on-chain media) not just a local algorithm.
- **Extends ISCC** – borrows semantic block coding ideas and hardens them with keyed, verifiable cryptographic structures.
- **Keyed Chaotic Coordinates** – logistic-map chaos with secret keys keeps sampling unpredictable.
- **Block SPN Hashing**: Implements a multi-round substitution-permutation network for robust block-level hashing
- **Compact Fingerprints**: Generates small JSON fingerprints that can verify image authenticity
- **High Performance**: Optimized for production use with configurable parameters
- **CLI Interface**: Easy-to-use command-line interface for fingerprint generation and verification
- **Configurable**: Flexible configuration system for tuning performance and security parameters

## 📑 Table of Contents
- [Installation](#installation)
- [Compare two images](#2-compare-two-images)
- [What do the numbers mean?](#what-do-the-numbers-mean)
- [Generate/Verify fingerprints](#3-generate-fingerprints--verify-images)
- [Configuration Options](#configuration-options)
- [API Usage](#api-usage)
- [Next Steps](#next-steps-for-enhancement)

---

## 🛠 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install BLAKE3 for enhanced security (recommended for production)
pip install blake3
```

## Quick Start

### 1. Generate a fingerprint

```bash
python protrace_spn.py gen --image path/to/image.png --image-id unique_id_123 --out fingerprint.json
```

### 2. Verify an image

```bash
python protrace_spn.py verify --image path/to/candidate.png --fingerprint fingerprint.json --out report.json
```

### 3. Create a custom configuration

```bash
python protrace_spn.py config --out custom_config.json
```

## Usage

### 1. Install dependencies
```bash
python -m pip install -r requirements.txt
```

### 2. Compare two images
Place (at least) two PNG images in the `images/` directory.  Simply run:
```bash
python diff_images.py
```
Without arguments the script automatically chooses the first two PNGs (alphabetical order) inside `images/` and prints a JSON report similar to:
```json
{
  "mse": 123.4,
  "psnr": 27.62,
  "percent_identical": 95.31,
  "percent_difference": 4.69,
  "diff_image": "images/diff_visual.png"
}
```
A visual diff (`diff_visual.png`) is also written next to the images.

#### What do the numbers mean?

| Field | Meaning |
|-------|---------|
| `mse` | Mean Squared Error – average squared pixel difference. 0 means identical. Lower is better. |
| `psnr` | Peak Signal-to-Noise Ratio in dB. ∞ for identical images. Higher is better (>40 dB ≈ nearly identical). |
| `percent_identical` | Percentage of pixels whose RGB values are exactly the same. |
| `percent_difference` | 100 − `percent_identical`. Easier to read when the images are mostly identical. |
| `diff_image` | Path to the amplified visual diff written by the script. White/bright pixels indicate where the two images differ. |

`diff_images.py` loads the two images, converts them to RGB, computes the error metrics above, saves an exaggerated difference visualization, and prints the JSON report. If you pass two filenames the same calculations are performed on that pair.

You can explicitly specify two paths:
```bash
python diff_images.py images/1.png images/2.png
```

### 3. Generate fingerprints / verify images
See `src/protrace/cli.py` for the full CLI:
```bash
python -m protrace.cli gen   --image images/1.png --image-id img1 --out fp1.json
python -m protrace.cli verify --image images/2.png --fingerprint fp1.json --out report.json
```
 Examples

### Basic Fingerprint Generation

```bash
# Generate fingerprint with default settings
python protrace_spn.py gen \
    --image sample.jpg \
    --image-id sample_001 \
    --out sample_fingerprint.json

# Generate with verbose logging
python protrace_spn.py gen \
    --image sample.jpg \
    --image-id sample_001 \
    --out sample_fingerprint.json \
    --verbose
```

### Custom Configuration

```bash
# Generate configuration file
python protrace_spn.py config --out production_config.json

# Edit the configuration file to adjust parameters
# Then use it for fingerprint generation
python protrace_spn.py gen \
    --image sample.jpg \
    --image-id sample_001 \
    --out sample_fingerprint.json \
    --config production_config.json
```

### Batch Processing

```python
import json
from protrace_spn import ProTraceConfig, ProTraceFingerprint

# Initialize with custom config
config = ProTraceConfig(
    block_size=32,
    num_chaotic_points=256,
    rounds=6,
    use_blake3=True
)

engine = ProTraceFingerprint(config)

# Process multiple images
images = ['img1.png', 'img2.png', 'img3.png']
for img_path in images:
    fingerprint = engine.generate_fingerprint(img_path, f"id_{img_path}")
    with open(f"{img_path}.fingerprint.json", 'w') as f:
        json.dump(fingerprint, f)
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `block_size` | 16 | Size of image blocks for hashing |
| `num_chaotic_points` | 128 | Number of chaotic sample points |
| `rounds` | 4 | Number of SPN rounds for hashing |
| `similarity_threshold` | 0.85 | Minimum similarity for verification |
| `use_blake3` | false | Use BLAKE3 instead of SHA256 |

## API Usage

```python
from protrace_spn import ProTraceConfig, ProTraceFingerprint

# Initialize
config = ProTraceConfig()
engine = ProTraceFingerprint(config)

# Generate fingerprint
fingerprint = engine.generate_fingerprint('image.png', 'unique_id')

# Verify image
report = engine.verify_image('candidate.png', fingerprint)

if report['is_verified']:
    print(f"Image verified with {report['overall_similarity']:.1%} similarity")
else:
    print("Image verification failed")
```

## Security Considerations

1. **Secret Key Management**: The system automatically generates and stores a secret key in `~/.protrace/secret.key`. This key should be:
   - Backed up securely
   - Protected with appropriate file permissions
   - Rotated periodically in production

2. **BLAKE3 for Production**: Enable BLAKE3 in production for better performance and security:
   ```json
   {
     "use_blake3": true
   }
   ```

3. **Threshold Tuning**: Adjust the similarity threshold based on your use case:
   - Higher threshold (>0.9): Stricter verification, fewer false positives
   - Lower threshold (<0.8): More lenient, allows for minor modifications

## Performance Optimization

### For Large Images
- Increase `block_size` to reduce the number of blocks
- Decrease `num_chaotic_points` for faster processing

### For High Security
- Increase `rounds` for stronger SPN hashing
- Increase `num_chaotic_points` for more sampling
- Enable BLAKE3 for faster checksums

### Parallel Processing
```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def process_image(args):
    img_path, image_id = args
    engine = ProTraceFingerprint(ProTraceConfig())
    return engine.generate_fingerprint(img_path, image_id)

# Process images in parallel
images = [('img1.png', 'id1'), ('img2.png', 'id2'), ('img3.png', 'id3')]
with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
    fingerprints = list(executor.map(process_image, images))
```

## Testing

Run the test suite:

```bash
python test_protrace.py
```

Run specific tests:

```bash
python -m unittest test_protrace.TestProTraceFingerprint
```

## Verification Report Format

The verification report includes:

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "image_id": "unique_id",
  "dimension_match": true,
  "chaotic_similarity": 0.95,
  "block_similarity": 0.92,
  "overall_similarity": 0.93,
  "is_verified": true,
  "threshold": 0.85,
  "details": {
    "chaotic_matches": "122/128",
    "block_matches": "230/250",
    "dimensions": "1920x1080 vs 1920x1080"
  }
}
```

## Production Deployment

### Docker Example

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY protrace_spn.py .

ENTRYPOINT ["python", "protrace_spn.py"]
```

### API Server Integration

```python
from fastapi import FastAPI, UploadFile
from protrace_spn import ProTraceConfig, ProTraceFingerprint

app = FastAPI()
engine = ProTraceFingerprint(ProTraceConfig())

@app.post("/fingerprint")
async def generate_fingerprint(file: UploadFile, image_id: str):
    # Save uploaded file temporarily
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    
    # Generate fingerprint
    fingerprint = engine.generate_fingerprint(temp_path, image_id)
    return fingerprint

@app.post("/verify")
async def verify_image(file: UploadFile, fingerprint: dict):
    # Similar implementation for verification
    pass
```

## Next Steps for Enhancement

1. **Hardening**: 
   - Integrate with HSM for key management
   - Implement epoch-based key rotation
   - Add support for distributed key management

2. **Performance**:
   - Compile critical paths to Rust/C extensions
   - Implement GPU acceleration for large-scale processing
   - Add caching layer for frequently accessed fingerprints

3. **Calibration**:
   - Run calibration suite across various image modifications
   - Auto-tune thresholds based on use case
   - Generate ROC curves for threshold selection

4. **API Development**:
   - Build REST API with FastAPI
   - Add WebSocket support for real-time verification
   - Implement batch processing endpoints

<!-- Proprietary project: no public license -->

## Contributing

Contributions are welcome! Please submit pull requests with:
- Unit tests for new features
- Documentation updates
- Performance benchmarks for optimizations

## Support

For issues and questions, please open an issue on the repository.
