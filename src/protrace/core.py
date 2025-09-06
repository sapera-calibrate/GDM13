"""
ProTrace SPN - Production-Ready Image Fingerprinting System
Generates compact fingerprints using keyed chaotic coordinates, block SPN hashes, and checksums.
Supports verification of candidate images against stored fingerprints.
Enhanced version: 
- Key-derived S-box and P-box for improved security.
- Added configurable sampler parameters.
- Improved performance with optional numba acceleration.
- Enhanced error handling and logging.
- Added support for batch processing in CLI.
- Removed legacy chaotic generator if not needed (commented out).
- Set use_blake3 to True by default for better performance/security.
- Added more documentation and type hints.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import numpy as np
from PIL import Image
import secrets
import hmac
from math import ceil
from scipy import ndimage

# Optional: For performance acceleration
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_BLOCK_SIZE = 16
DEFAULT_NUM_CHAOTIC_POINTS = 128
DEFAULT_ROUNDS = 4
DEFAULT_SIMILARITY_THRESHOLD = 0.85
DEFAULT_F_GRID = 0.25
DEFAULT_B_BLOCK = 32
DEFAULT_IMPORTANCE_METHOD = "sobel"
VERSION = "1.1.0"  # Updated version for enhancements

class ConfigError(Exception):
    """Configuration related errors"""
    pass

class FingerprintError(Exception):
    """Fingerprint generation/verification errors"""
    pass

@dataclass
class ProTraceConfig:
    """Configuration for ProTrace fingerprinting"""
    block_size: int = DEFAULT_BLOCK_SIZE
    num_chaotic_points: int = DEFAULT_NUM_CHAOTIC_POINTS
    rounds: int = DEFAULT_ROUNDS
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    use_blake3: bool = True  # Enhanced: Default to True for production
    secret_key: Optional[bytes] = None
    f_grid: float = DEFAULT_F_GRID  # Enhanced: Configurable sampler params
    b_block: int = DEFAULT_B_BLOCK
    importance_method: str = DEFAULT_IMPORTANCE_METHOD
    
    @classmethod
    def from_json(cls, path: str) -> 'ProTraceConfig':
        """Load configuration from JSON file"""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return cls(**data)
        except Exception as e:
            raise ConfigError(f"Failed to load config from {path}: {e}")
    
    def to_json(self, path: str):
        """Save configuration to JSON file"""
        data = asdict(self)
        # Don't save secret key to file
        data.pop('secret_key', None)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

def derive_seed(secret: bytes, image_id: str, epoch: str, salt=b"protrace-dither") -> int:
    """Derive deterministic seed from secret key and image ID"""
    mac = hmac.new(secret, (image_id + "|" + epoch).encode('utf-8') + salt, hashlib.sha256).digest()
    seed = int.from_bytes(mac[:8], 'big') & 0x7fffffffffffffff
    return seed

@njit
def compute_block_energy_numba(imgp: np.ndarray) -> np.ndarray:
    """Numba-accelerated Sobel energy computation"""
    H, W = imgp.shape
    mag = np.zeros((H, W), dtype=np.float64)
    for i in prange(1, H-1):
        for j in prange(1, W-1):
            sx = imgp[i, j+1] - imgp[i, j-1]
            sy = imgp[i+1, j] - imgp[i-1, j]
            mag[i, j] = np.sqrt(sx**2 + sy**2)
    return mag

def compute_block_energy(img_gray: np.ndarray, B: int = 32, method: str = "sobel") -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """Compute ISCC-like block energy map with optional numba acceleration"""
    H, W = img_gray.shape
    Bh = ceil(H / B)
    Bw = ceil(W / B)
    # Pad image to multiple of B for simplicity
    padH = Bh * B - H
    padW = Bw * B - W
    imgp = np.pad(img_gray, ((0, padH), (0, padW)), mode='reflect')
    
    if method == "sobel":
        if HAS_NUMBA:
            mag = compute_block_energy_numba(imgp.astype(np.float64))
        else:
            sx = ndimage.sobel(imgp.astype(float), axis=1)
            sy = ndimage.sobel(imgp.astype(float), axis=0)
            mag = np.hypot(sx, sy)
    elif method == "dct":
        # Approximate energy by local variance as cheap proxy
        mag = ndimage.gaussian_filter(imgp.astype(float), sigma=1, mode='reflect')
        mag = np.abs(imgp.astype(float) - mag)
    else:
        mag = ndimage.gaussian_filter(imgp.astype(float), sigma=1)
    
    # Reduce to block energy
    energy = mag.reshape(Bh, B, Bw, B).sum(axis=(1, 3))
    return energy, (Bh, Bw), (padH, padW)

def grid_baseline_coords(H: int, W: int, grid_count: int, rng: np.random.Generator) -> List[Tuple[int, int]]:
    """Generate grid baseline coordinates with jitter"""
    g_rows = max(1, int(np.floor(np.sqrt(grid_count * H / W))))
    g_cols = int(ceil(grid_count / g_rows))
    cell_h = H / g_rows
    cell_w = W / g_cols
    coords = []
    for r in range(g_rows):
        for c in range(g_cols):
            jr = int(min(H-1, max(0, (r + rng.random()) * cell_h)))
            jc = int(min(W-1, max(0, (c + rng.random()) * cell_w)))
            coords.append((jr, jc))
    return coords

@njit
def floyd_steinberg_block_dither_numba(importance_block: np.ndarray, samples_to_emit: int) -> List[Tuple[int, int]]:
    """Numba-accelerated Floyd-Steinberg dithering within a block"""
    Hb, Wb = importance_block.shape
    target = importance_block * (samples_to_emit / (importance_block.sum() + 1e-12))
    out = np.zeros((Hb, Wb), dtype=np.uint8)
    err = target.copy()
    for y in range(Hb):
        for x in range(Wb):
            v = err[y, x]
            if v >= 0.5:
                out[y, x] = 1
                quant = 1.0
            else:
                out[y, x] = 0
                quant = 0.0
            e = v - quant
            if x + 1 < Wb:
                err[y, x + 1] += e * 7 / 16
            if y + 1 < Hb:
                if x - 1 >= 0:
                    err[y + 1, x - 1] += e * 3 / 16
                err[y + 1, x] += e * 5 / 16
                if x + 1 < Wb:
                    err[y + 1, x + 1] += e * 1 / 16
    pts = []
    for y in range(Hb):
        for x in range(Wb):
            if out[y, x] == 1:
                pts.append((y, x))
    # Note: Shuffling and residual filling not accelerated; handle outside if needed
    return pts

def floyd_steinberg_block_dither(importance_block: np.ndarray, samples_to_emit: int, rng: np.random.Generator) -> List[Tuple[int, int]]:
    """Error-diffusion dithering within a block with optional numba"""
    if HAS_NUMBA:
        pts = floyd_steinberg_block_dither_numba(importance_block, samples_to_emit)
    else:
        # Original implementation...
        Hb, Wb = importance_block.shape
        target = importance_block * (samples_to_emit / (importance_block.sum() + 1e-12))
        out = np.zeros_like(importance_block, dtype=np.uint8)
        err = target.copy()
        for y in range(Hb):
            for x in range(Wb):
                v = err[y, x]
                if v >= 0.5:
                    out[y, x] = 1
                    quant = 1.0
                else:
                    out[y, x] = 0
                    quant = 0.0
                e = v - quant
                if x + 1 < Wb: err[y, x + 1] += e * 7/16
                if y + 1 < Hb:
                    if x - 1 >= 0: err[y + 1, x - 1] += e * 3/16
                    err[y + 1, x] += e * 5/16
                    if x + 1 < Wb: err[y + 1, x + 1] += e * 1/16
        pts = list(zip(*np.where(out == 1)))

    if len(pts) > samples_to_emit:
        rng.shuffle(pts)
        pts = pts[:samples_to_emit]
    elif len(pts) < samples_to_emit:
        residuals = (importance_block * (samples_to_emit / (importance_block.sum() + 1e-12)) - out).ravel()
        idx_sorted = np.argsort(-residuals)
        need = samples_to_emit - len(pts)
        flat_coords = [(i // importance_block.shape[1], i % importance_block.shape[1]) for i in idx_sorted[:need]]
        pts.extend(flat_coords)
    rng.shuffle(pts)
    return pts

def hybrid_dither_sampler(image_path: str, image_id: str, secret: bytes, epoch: str = "v1", M_total: int = 4096, f_grid: float = 0.25, B_block: int = 32, importance_method: str = "sobel") -> List[Tuple[int, int]]:
    """Hybrid dithering + ISCC-style sampler"""
    img = Image.open(image_path).convert("L")
    arr = np.array(img, dtype=np.uint8)
    H, W = arr.shape
    rng = np.random.default_rng(derive_seed(secret, image_id, epoch))
    
    # 1) grid baseline
    M_grid = int(round(M_total * f_grid))
    grid_coords = grid_baseline_coords(H, W, M_grid, rng)
    
    # 2) ISCC-like block energy map
    energy, (Bh, Bw), _ = compute_block_energy(arr, B=B_block, method=importance_method)
    flat_energy = energy / (energy.sum() + 1e-12)  # normalize
    
    # target per-block samples from remaining budget
    remaining = M_total - len(grid_coords)
    block_targets = (flat_energy * remaining).astype(int)
    
    # ensure sum equals remaining by distributing residuals
    deficit = remaining - block_targets.sum()
    if deficit > 0:
        flat_idx = np.argsort(-flat_energy.ravel())
        for k in flat_idx[:deficit]:
            i = k // Bw
            j = k % Bw
            block_targets[i, j] += 1
    
    # 3) For each block, run dithering to choose in-block pixels
    sampled_coords = list(grid_coords)
    for bi in range(Bh):
        for bj in range(Bw):
            samples_in_block = int(block_targets[bi, bj])
            if samples_in_block <= 0:
                continue
            
            # extract block pixel importance
            r0 = bi * B_block
            c0 = bj * B_block
            block = arr[r0:r0+B_block, c0:c0+B_block].astype(float)
            
            # compute simple per-pixel energy
            gx = ndimage.sobel(block, axis=1, mode='reflect')
            gy = ndimage.sobel(block, axis=0, mode='reflect')
            block_energy = np.hypot(gx, gy)
            
            # normalize and call dithering selection
            if block_energy.sum() <= 1e-9:
                # fallback: uniform random within block
                choices = [(int(rng.random() * block.shape[0]), int(rng.random() * block.shape[1])) for _ in range(samples_in_block)]
            else:
                norm_be = block_energy / (block_energy.sum() + 1e-12)
                choices = floyd_steinberg_block_dither(norm_be, samples_in_block, rng)
            
            # map block-local coords to global
            for rr, cc in choices:
                global_r = min(H - 1, r0 + rr)
                global_c = min(W - 1, c0 + cc)
                sampled_coords.append((global_r, global_c))
    
    # 4) If still short, add chaotic fill
    while len(sampled_coords) < M_total:
        i = int(rng.integers(0, H))
        j = int(rng.integers(0, W))
        sampled_coords.append((i, j))
    
    # 5) deterministic keyed shuffle for unpredictability
    perm_rng = np.random.default_rng(derive_seed(secret, image_id, epoch + "|perm"))
    perm = perm_rng.permutation(len(sampled_coords))
    sampled_coords = [sampled_coords[i] for i in perm]
    
    # 6) trim to M_total
    return sampled_coords[:M_total]

# Legacy ChaoticGenerator commented out for enhancement; remove if backward compatibility not needed
# class ChaoticGenerator:
#     ...

class SPNHash:
    """Substitution-Permutation Network for block hashing - Enhanced with key-derived S/P boxes"""
    
    def __init__(self, rounds: int = 4, secret_key: Optional[bytes] = None):
        self.rounds = rounds
        self.secret_key = secret_key
        self._init_sbox()
        self._init_pbox()
    
    def _init_sbox(self):
        """Initialize S-box for substitution, derived from key if provided"""
        self.sbox = np.arange(256, dtype=np.uint8)
        if self.secret_key:
            seed = int.from_bytes(hashlib.sha256(self.secret_key + b"sbox").digest()[:8], 'big')
        else:
            seed = 42
        np.random.seed(seed)
        np.random.shuffle(self.sbox)
    
    def _init_pbox(self):
        """Initialize P-box for permutation, derived from key if provided"""
        self.pbox = np.arange(128, dtype=np.uint8)
        if self.secret_key:
            seed = int.from_bytes(hashlib.sha256(self.secret_key + b"pbox").digest()[:8], 'big')
        else:
            seed = 43
        np.random.seed(seed)
        np.random.shuffle(self.pbox)
    
    def hash_block(self, block: np.ndarray, key: bytes) -> bytes:
        """Hash a block using SPN"""
        data = block.flatten().astype(np.uint8)
        if len(data) < 128:
            data = np.pad(data, (0, 128 - len(data)), mode='constant')
        elif len(data) > 128:
            data = data[:128]
        
        round_keys = self._derive_round_keys(key, self.rounds)
        
        for round_idx in range(self.rounds):
            data = data ^ round_keys[round_idx]
            data = self.sbox[data]
            if round_idx < self.rounds - 1:
                data = self._permute(data)
        
        data = data ^ round_keys[-1]
        return data.tobytes()[:32]
    
    def _derive_round_keys(self, key: bytes, num_rounds: int) -> List[np.ndarray]:
        keys = []
        for i in range(num_rounds + 1):
            round_key = hashlib.sha256(key + i.to_bytes(4, 'little')).digest()
            keys.append(np.frombuffer(round_key * 4, dtype=np.uint8)[:128])
        return keys
    
    def _permute(self, data: np.ndarray) -> np.ndarray:
        return data[self.pbox]
    
    def hash_block_to_bits(self, block: np.ndarray, key: bytes) -> np.ndarray:
        hash_bytes = self.hash_block(block, key)
        bits = np.unpackbits(np.frombuffer(hash_bytes, dtype=np.uint8))
        return bits.astype(np.uint8)

def compute_verification_report(fingerprint: Dict[str, Any],
                                candidate_block_hashes: List[np.ndarray],
                                candidate_checksums: Dict[int, int],
                                ensemble_results: Optional[Dict[str, Any]] = None,
                                verified_threshold: float = 0.90,
                                per_block_flag_threshold: float = 0.12,
                                checksum_mismatch_tolerance: float = 0.02) -> Dict[str, Any]:
    """
    Build robust verification report.
    Enhanced: Added type hints and minor optimizations.
    """
    B = fingerprint["B"]
    stored_bits_list = []
    for s in fingerprint["block_hashes"]:
        if isinstance(s, str):
            arr = np.array([int(ch) for ch in s], dtype=np.uint8)
        elif isinstance(s, dict) and 'bits' in s:
            arr = np.array(s['bits'], dtype=np.uint8)
        else:
            arr = np.array(s, dtype=np.uint8)
        stored_bits_list.append(arr)

    per_block_f = []
    per_block_similarity = []
    for i in range(B):
        stored_bits = stored_bits_list[i]
        cand_bits = candidate_block_hashes[i]
        mismatches = np.sum(stored_bits != cand_bits)
        f = mismatches / stored_bits.size
        per_block_f.append(f)
        per_block_similarity.append(max(0.0, 1.0 - f))

    avg_block_similarity = np.mean(per_block_similarity)

    stored_checks = fingerprint.get("tiny_checksums", {})
    mismatches = sum(1 for idx, stored_val in stored_checks.items() if int(candidate_checksums.get(int(idx), -1)) != int(stored_val))
    total_checks = max(1, len(stored_checks))
    mismatch_fraction = mismatches / total_checks
    checksum_similarity = 1.0 - mismatch_fraction

    if ensemble_results:
        ens_sims = [np.mean([max(0.0, 1.0 - x) for x in r["per_block"]]) for r in ensemble_results.values()]
        if ens_sims:
            ensemble_mean_sim = np.mean(ens_sims)
            avg_block_similarity = 0.6 * avg_block_similarity + 0.4 * ensemble_mean_sim

    GLOBAL_WEIGHT_BLOCK = 0.8
    GLOBAL_WEIGHT_CHECKSUM = 0.2
    global_similarity = GLOBAL_WEIGHT_BLOCK * avg_block_similarity + GLOBAL_WEIGHT_CHECKSUM * checksum_similarity

    flagged_blocks = [i for i, f in enumerate(per_block_f) if f > per_block_flag_threshold]

    is_verified = (global_similarity >= verified_threshold) and (mismatch_fraction <= checksum_mismatch_tolerance) and not flagged_blocks

    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "image_id": fingerprint.get("image_id"),
        "dimension_match": True,
        "avg_block_similarity": float(avg_block_similarity),
        "checksum_similarity": float(checksum_similarity),
        "global_similarity": float(global_similarity),
        "is_verified": bool(is_verified),
        "verified_threshold": verified_threshold,
        "per_block_similarity": [float(s) for s in per_block_similarity],
        "per_block_hamming": [float(f) for f in per_block_f],
        "flagged_blocks": flagged_blocks,
        "tiny_checks_total": total_checks,
        "tiny_checks_mismatches": mismatches,
        "tiny_mismatch_fraction": float(mismatch_fraction)
    }
    return report

class ProTraceFingerprint:
    """Main fingerprinting engine - Enhanced with key-derived SPN"""
    
    def __init__(self, config: ProTraceConfig):
        self.config = config
        self.spn = SPNHash(config.rounds, config.secret_key)
        if config.secret_key is None:
            config.secret_key = self._get_or_create_key()
    
    def _get_or_create_key(self) -> bytes:
        """Get or create a secret key for fingerprinting"""
        key_path = Path.home() / '.protrace' / 'secret.key'
        key_path.parent.mkdir(parents=True, exist_ok=True)
        if key_path.exists():
            with open(key_path, 'rb') as f:
                return f.read()
        key = secrets.token_bytes(32)
        with open(key_path, 'wb') as f:
            f.write(key)
        logger.info(f"Created new secret key at {key_path}")
        return key
    
    def generate_fingerprint(self, image_path: str, image_id: str) -> Dict[str, Any]:
        """Generate a fingerprint for an image"""
        try:
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_array = np.array(img)
            height, width = img_array.shape[:2]
            
            epoch = "v2_hybrid"
            sampled_coords = hybrid_dither_sampler(
                image_path=image_path,
                image_id=image_id,
                secret=self.config.secret_key,
                epoch=epoch,
                M_total=self.config.num_chaotic_points,
                f_grid=self.config.f_grid,
                B_block=self.config.b_block,
                importance_method=self.config.importance_method
            )
            
            chaotic_samples = []
            for y, x in sampled_coords:
                sample = img_array[y, x].tolist()
                chaotic_samples.append({'coord': [x, y], 'value': sample})
            
            block_hashes = []
            block_size = self.config.block_size

            block_epoch = epoch + "_blocks"
            block_sampled_coords = hybrid_dither_sampler(
                image_path=image_path,
                image_id=image_id + "_blocks",
                secret=self.config.secret_key,
                epoch=block_epoch,
                M_total=self.config.num_chaotic_points * 2,
                f_grid=self.config.f_grid * 0.6,  # Adjusted for blocks
                B_block=self.config.b_block // 2,
                importance_method=self.config.importance_method
            )
            
            block_positions = []
            target_count = max(64, (height // (block_size * 2)) * (width // (block_size * 2)))

            for y, x in block_sampled_coords:
                i = y - (y % block_size)
                j = x - (x % block_size)
                if i + block_size <= height and j + block_size <= width:
                    block_positions.append((i, j))
                if len(block_positions) >= target_count:
                    break

            seen = set()
            unique_positions = [pos for pos in block_positions if pos not in seen and not seen.add(pos)]

            for i, j in unique_positions:
                block = img_array[i:i+block_size, j:j+block_size]
                block_key = self.config.secret_key + f'{i}_{j}_{image_id}'.encode()
                block_bits = self.spn.hash_block_to_bits(block, block_key)
                block_hashes.append({
                    'position': [i, j],
                    'bits': block_bits.tolist(),
                    'hash': self.spn.hash_block(block, block_key).hex()
                })
            
            tiny_checksums = {}
            for idx, sample in enumerate(chaotic_samples):
                if idx % 8 == 0:
                    r, g, b = sample['value']
                    checksum = (r ^ g ^ b) & 0xFF
                    tiny_checksums[str(idx)] = checksum
            
            if self.config.use_blake3:
                try:
                    import blake3
                    hasher = blake3.blake3()
                except ImportError:
                    logger.warning("BLAKE3 not available, falling back to SHA256")
                    hasher = hashlib.sha256()
            else:
                hasher = hashlib.sha256()
            
            hasher.update(img_array.tobytes())
            checksum = hasher.hexdigest()
            
            fingerprint = {
                'version': VERSION,
                'image_id': image_id,
                'timestamp': datetime.utcnow().isoformat(),
                'dimensions': [width, height],
                'checksum': checksum,
                'chaotic_samples': chaotic_samples,
                'block_hashes': block_hashes,
                'tiny_checksums': tiny_checksums,
                'B': len(block_hashes),
                'sampler': {
                    'type': 'hybrid_dither',
                    'epoch': epoch,
                    'f_grid': self.config.f_grid,
                    'B_block': self.config.b_block,
                    'importance_method': self.config.importance_method,
                    'M_total': self.config.num_chaotic_points
                },
                'config': {
                    'block_size': self.config.block_size,
                    'num_chaotic_points': self.config.num_chaotic_points,
                    'rounds': self.config.rounds
                }
            }
            return fingerprint
        except FileNotFoundError as e:
            raise FingerprintError(f"Image file not found: {image_path}") from e
        except Exception as e:
            raise FingerprintError(f"Failed to generate fingerprint: {str(e)}") from e
    
    def verify_image(self, image_path: str, fingerprint: Dict[str, Any]) -> Dict[str, Any]:
        """Verify an image against a fingerprint using Hamming-based similarity"""
        try:
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_array = np.array(img)
            height, width = img_array.shape[:2]
            
            expected_dims = fingerprint['dimensions']
            dimension_match = (width == expected_dims[0] and height == expected_dims[1])
            
            if not dimension_match:
                return {
                    'timestamp': datetime.utcnow().isoformat() + 'Z',
                    'image_id': fingerprint['image_id'],
                    'dimension_match': False,
                    'global_similarity': 0.0,
                    'is_verified': False,
                    'error': f"Dimension mismatch: {width}x{height} vs {expected_dims[0]}x{expected_dims[1]}"
                }
            
            candidate_block_hashes = []
            block_size = fingerprint['config']['block_size']
            fingerprint_image_id = fingerprint['image_id']
            
            for block_info in fingerprint['block_hashes']:
                i, j = block_info['position']
                if i + block_size <= height and j + block_size <= width:
                    block = img_array[i:i+block_size, j:j+block_size]
                    block_key = self.config.secret_key + f'{i}_{j}_{fingerprint_image_id}'.encode()
                    block_bits = self.spn.hash_block_to_bits(block, block_key)
                else:
                    block_bits = np.zeros(256, dtype=np.uint8)
                candidate_block_hashes.append(block_bits)
            
            candidate_checksums = {}
            for idx, sample in enumerate(fingerprint['chaotic_samples']):
                if idx % 8 == 0:
                    x, y = sample['coord']
                    if 0 <= x < width and 0 <= y < height:
                        r, g, b = img_array[y, x]
                        checksum = (r ^ g ^ b) & 0xFF
                        candidate_checksums[idx] = checksum
            
            report = compute_verification_report(
                fingerprint=fingerprint,
                candidate_block_hashes=candidate_block_hashes,
                candidate_checksums=candidate_checksums,
                verified_threshold=0.90,
                per_block_flag_threshold=0.12,
                checksum_mismatch_tolerance=0.02
            )
            
            report['dimension_match'] = dimension_match
            report['chaotic_similarity'] = report['checksum_similarity']
            report['block_similarity'] = report['avg_block_similarity']
            report['overall_similarity'] = report['global_similarity']
            report['threshold'] = report['verified_threshold']
            
            return report
        except FileNotFoundError as e:
            raise FingerprintError(f"Image file not found: {image_path}") from e
        except Exception as e:
            raise FingerprintError(f"Failed to verify image: {str(e)}") from e

class ProTraceCLI:
    """Command-line interface for ProTrace - Enhanced with batch support"""
    
    def __init__(self):
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description='ProTrace SPN - Image Fingerprinting System',
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        parser.add_argument('--version', action='version', version=f'ProTrace v{VERSION}')
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Generate fingerprint command
        gen_parser = subparsers.add_parser('gen', help='Generate fingerprint for an image or batch')
        gen_parser.add_argument('--image', help='Path to input image (single)')
        gen_parser.add_argument('--batch', help='Path to JSON list of {"image": path, "image_id": id}')
        gen_parser.add_argument('--image-id', help='Unique identifier for the image (single)')
        gen_parser.add_argument('--out', required=True, help='Output path for fingerprint JSON or directory for batch')
        gen_parser.add_argument('--config', help='Path to configuration file')
        gen_parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
        
        # Verify image command
        verify_parser = subparsers.add_parser('verify', help='Verify image against fingerprint or batch')
        verify_parser.add_argument('--image', help='Path to candidate image (single)')
        verify_parser.add_argument('--batch', help='Path to JSON list of {"image": path, "fingerprint": path}')
        verify_parser.add_argument('--fingerprint', help='Path to fingerprint JSON (single)')
        verify_parser.add_argument('--out', required=True, help='Output path for verification report or directory for batch')
        verify_parser.add_argument('--config', help='Path to configuration file')
        verify_parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
        
        # Config command
        config_parser = subparsers.add_parser('config', help='Generate default configuration file')
        config_parser.add_argument('--out', required=True, help='Output path for configuration file')
        
        return parser
    
    def run(self, args=None):
        args = self.parser.parse_args(args)
        
        if not args.command:
            self.parser.print_help()
            return 1
        
        if hasattr(args, 'verbose') and args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        try:
            if args.command == 'gen':
                return self._generate_fingerprint(args)
            elif args.command == 'verify':
                return self._verify_image(args)
            elif args.command == 'config':
                return self._generate_config(args)
        except Exception as e:
            logger.error(f"Error: {e}")
            return 1
    
    def _generate_fingerprint(self, args) -> int:
        if args.config:
            config = ProTraceConfig.from_json(args.config)
        else:
            config = ProTraceConfig()
        
        engine = ProTraceFingerprint(config)
        
        if args.batch:
            with open(args.batch, 'r') as f:
                batch_list = json.load(f)
            os.makedirs(args.out, exist_ok=True)
            for item in batch_list:
                image_path = item['image']
                image_id = item['image_id']
                logger.info(f"Generating fingerprint for {image_path}")
                start_time = time.time()
                fingerprint = engine.generate_fingerprint(image_path, image_id)
                elapsed = time.time() - start_time
                out_path = os.path.join(args.out, f"{image_id}.json")
                with open(out_path, 'w') as f:
                    json.dump(fingerprint, f, indent=2)
                logger.info(f"Fingerprint saved to {out_path} (took {elapsed:.2f}s)")
        elif args.image and args.image_id:
            logger.info(f"Generating fingerprint for {args.image}")
            start_time = time.time()
            fingerprint = engine.generate_fingerprint(args.image, args.image_id)
            elapsed = time.time() - start_time
            with open(args.out, 'w') as f:
                json.dump(fingerprint, f, indent=2)
            logger.info(f"Fingerprint saved to {args.out} (took {elapsed:.2f}s)")
        else:
            logger.error("Must provide --image and --image-id or --batch")
            return 1
        return 0
    
    def _verify_image(self, args) -> int:
        if args.config:
            config = ProTraceConfig.from_json(args.config)
        else:
            config = ProTraceConfig()
        
        engine = ProTraceFingerprint(config)
        
        if args.batch:
            with open(args.batch, 'r') as f:
                batch_list = json.load(f)
            os.makedirs(args.out, exist_ok=True)
            for item in batch_list:
                image_path = item['image']
                fp_path = item['fingerprint']
                with open(fp_path, 'r') as f:
                    fingerprint = json.load(f)
                if 'config' in fingerprint:
                    config.block_size = fingerprint['config'].get('block_size', config.block_size)
                    config.num_chaotic_points = fingerprint['config'].get('num_chaotic_points', config.num_chaotic_points)
                    config.rounds = fingerprint['config'].get('rounds', config.rounds)
                logger.info(f"Verifying {image_path} against {fp_path}")
                start_time = time.time()
                report = engine.verify_image(image_path, fingerprint)
                elapsed = time.time() - start_time
                out_path = os.path.join(args.out, f"{fingerprint['image_id']}_report.json")
                with open(out_path, 'w') as f:
                    json.dump(report, f, indent=2)
                logger.info(f"Verification report saved to {out_path} (took {elapsed:.2f}s)")
                if report['is_verified']:
                    logger.info("✓ Image VERIFIED")
                else:
                    logger.warning("✗ Image NOT VERIFIED")
        elif args.image and args.fingerprint:
            with open(args.fingerprint, 'r') as f:
                fingerprint = json.load(f)
            if 'config' in fingerprint:
                config.block_size = fingerprint['config'].get('block_size', config.block_size)
                config.num_chaotic_points = fingerprint['config'].get('num_chaotic_points', config.num_chaotic_points)
                config.rounds = fingerprint['config'].get('rounds', config.rounds)
            logger.info(f"Verifying {args.image} against fingerprint")
            start_time = time.time()
            report = engine.verify_image(args.image, fingerprint)
            elapsed = time.time() - start_time
            with open(args.out, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Verification completed in {elapsed:.2f}s")
            if report['is_verified']:
                logger.info("✓ Image VERIFIED")
            else:
                logger.warning("✗ Image NOT VERIFIED")
            return 0 if report['is_verified'] else 2
        else:
            logger.error("Must provide --image and --fingerprint or --batch")
            return 1
        return 0
    
    def _generate_config(self, args) -> int:
        config = ProTraceConfig()
        config.to_json(args.out)
        logger.info(f"Default configuration saved to {args.out}")
        return 0

def main():
    cli = ProTraceCLI()
    sys.exit(cli.run())

if __name__ == '__main__':
    main()