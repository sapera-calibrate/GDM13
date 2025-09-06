#!/usr/bin/env python3
"""
Unit tests for ProTrace SPN fingerprinting system
"""

import unittest
import tempfile
import json
import os
import numpy as np
from PIL import Image
from pathlib import Path
import shutil

from protrace_spn import (
    ProTraceConfig,
    ProTraceFingerprint,
    ChaoticGenerator,
    SPNHash,
    FingerprintError
)

class TestProTraceConfig(unittest.TestCase):
    """Test configuration handling"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = ProTraceConfig()
        self.assertEqual(config.block_size, 16)
        self.assertEqual(config.num_chaotic_points, 128)
        self.assertEqual(config.rounds, 4)
        self.assertAlmostEqual(config.similarity_threshold, 0.85)
        self.assertFalse(config.use_blake3)
    
    def test_config_serialization(self):
        """Test saving and loading configuration"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
        
        try:
            # Save config
            config = ProTraceConfig(block_size=32, num_chaotic_points=256)
            config.to_json(config_path)
            
            # Load config
            loaded_config = ProTraceConfig.from_json(config_path)
            self.assertEqual(loaded_config.block_size, 32)
            self.assertEqual(loaded_config.num_chaotic_points, 256)
        finally:
            os.unlink(config_path)

class TestChaoticGenerator(unittest.TestCase):
    """Test chaotic coordinate generation"""
    
    def test_point_generation(self):
        """Test that chaotic points are within image bounds"""
        seed = b"test_seed_123"
        width, height = 640, 480
        gen = ChaoticGenerator(seed, width, height)
        
        points = gen.generate_points(100)
        self.assertEqual(len(points), 100)
        
        for x, y in points:
            self.assertGreaterEqual(x, 0)
            self.assertLess(x, width)
            self.assertGreaterEqual(y, 0)
            self.assertLess(y, height)
    
    def test_deterministic_generation(self):
        """Test that same seed produces same points"""
        seed = b"test_seed_456"
        width, height = 800, 600
        
        gen1 = ChaoticGenerator(seed, width, height)
        points1 = gen1.generate_points(50)
        
        gen2 = ChaoticGenerator(seed, width, height)
        points2 = gen2.generate_points(50)
        
        self.assertEqual(points1, points2)
    
    def test_different_seeds(self):
        """Test that different seeds produce different points"""
        width, height = 800, 600
        
        gen1 = ChaoticGenerator(b"seed1", width, height)
        points1 = gen1.generate_points(50)
        
        gen2 = ChaoticGenerator(b"seed2", width, height)
        points2 = gen2.generate_points(50)
        
        self.assertNotEqual(points1, points2)

class TestSPNHash(unittest.TestCase):
    """Test SPN hashing functionality"""
    
    def test_hash_consistency(self):
        """Test that same block produces same hash"""
        spn = SPNHash(rounds=4)
        key = b"test_key_789"
        
        # Create a test block
        block = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
        
        hash1 = spn.hash_block(block, key)
        hash2 = spn.hash_block(block, key)
        
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 32)  # 32-byte hash
    
    def test_hash_sensitivity(self):
        """Test that different blocks produce different hashes"""
        spn = SPNHash(rounds=4)
        key = b"test_key_abc"
        
        block1 = np.zeros((16, 16, 3), dtype=np.uint8)
        block2 = np.ones((16, 16, 3), dtype=np.uint8)
        
        hash1 = spn.hash_block(block1, key)
        hash2 = spn.hash_block(block2, key)
        
        self.assertNotEqual(hash1, hash2)

class TestProTraceFingerprint(unittest.TestCase):
    """Test fingerprinting functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Create test images"""
        cls.test_dir = tempfile.mkdtemp()
        
        # Create test image 1
        img1 = Image.new('RGB', (256, 256), color=(100, 150, 200))
        cls.test_image1 = os.path.join(cls.test_dir, 'test1.png')
        img1.save(cls.test_image1)
        
        # Create test image 2 (different)
        img2 = Image.new('RGB', (256, 256), color=(200, 100, 50))
        cls.test_image2 = os.path.join(cls.test_dir, 'test2.png')
        img2.save(cls.test_image2)
        
        # Create test image 3 (slightly modified version of image 1)
        img3 = img1.copy()
        pixels = img3.load()
        for i in range(10):
            for j in range(10):
                pixels[i, j] = (150, 150, 150)
        cls.test_image3 = os.path.join(cls.test_dir, 'test3.png')
        img3.save(cls.test_image3)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test images"""
        shutil.rmtree(cls.test_dir)
    
    def test_fingerprint_generation(self):
        """Test fingerprint generation"""
        config = ProTraceConfig()
        engine = ProTraceFingerprint(config)
        
        fingerprint = engine.generate_fingerprint(self.test_image1, "test_id_1")
        
        # Check fingerprint structure
        self.assertIn('version', fingerprint)
        self.assertIn('image_id', fingerprint)
        self.assertIn('timestamp', fingerprint)
        self.assertIn('dimensions', fingerprint)
        self.assertIn('checksum', fingerprint)
        self.assertIn('chaotic_samples', fingerprint)
        self.assertIn('block_hashes', fingerprint)
        
        self.assertEqual(fingerprint['image_id'], 'test_id_1')
        self.assertEqual(fingerprint['dimensions'], [256, 256])
        self.assertEqual(len(fingerprint['chaotic_samples']), config.num_chaotic_points)
        self.assertGreater(len(fingerprint['block_hashes']), 0)
    
    def test_verification_same_image(self):
        """Test verification with same image"""
        config = ProTraceConfig()
        engine = ProTraceFingerprint(config)
        
        # Generate fingerprint
        fingerprint = engine.generate_fingerprint(self.test_image1, "test_id_2")
        
        # Verify same image
        report = engine.verify_image(self.test_image1, fingerprint)
        
        self.assertTrue(report['is_verified'])
        self.assertTrue(report['dimension_match'])
        self.assertAlmostEqual(report['chaotic_similarity'], 1.0)
        self.assertAlmostEqual(report['block_similarity'], 1.0)
        self.assertAlmostEqual(report['overall_similarity'], 1.0)
    
    def test_verification_different_image(self):
        """Test verification with completely different image"""
        config = ProTraceConfig()
        engine = ProTraceFingerprint(config)
        
        # Generate fingerprint for image 1
        fingerprint = engine.generate_fingerprint(self.test_image1, "test_id_3")
        
        # Verify with different image
        report = engine.verify_image(self.test_image2, fingerprint)
        
        self.assertFalse(report['is_verified'])
        self.assertLess(report['overall_similarity'], config.similarity_threshold)
    
    def test_verification_modified_image(self):
        """Test verification with slightly modified image"""
        config = ProTraceConfig(similarity_threshold=0.7)
        engine = ProTraceFingerprint(config)
        
        # Generate fingerprint for original
        fingerprint = engine.generate_fingerprint(self.test_image1, "test_id_4")
        
        # Verify with slightly modified image
        report = engine.verify_image(self.test_image3, fingerprint)
        
        # Should have high but not perfect similarity
        self.assertTrue(report['dimension_match'])
        self.assertGreater(report['overall_similarity'], 0.5)
        self.assertLess(report['overall_similarity'], 1.0)
    
    def test_fingerprint_persistence(self):
        """Test saving and loading fingerprints"""
        config = ProTraceConfig()
        engine = ProTraceFingerprint(config)
        
        # Generate fingerprint
        fingerprint = engine.generate_fingerprint(self.test_image1, "test_id_5")
        
        # Save to file
        fingerprint_path = os.path.join(self.test_dir, 'fingerprint.json')
        with open(fingerprint_path, 'w') as f:
            json.dump(fingerprint, f)
        
        # Load from file
        with open(fingerprint_path, 'r') as f:
            loaded_fingerprint = json.load(f)
        
        # Verify with loaded fingerprint
        report = engine.verify_image(self.test_image1, loaded_fingerprint)
        self.assertTrue(report['is_verified'])

class TestErrorHandling(unittest.TestCase):
    """Test error handling"""
    
    def test_invalid_image_path(self):
        """Test handling of invalid image path"""
        config = ProTraceConfig()
        engine = ProTraceFingerprint(config)
        
        with self.assertRaises(FingerprintError):
            engine.generate_fingerprint("nonexistent.png", "test_id")
    
    def test_invalid_fingerprint_format(self):
        """Test handling of invalid fingerprint format"""
        config = ProTraceConfig()
        engine = ProTraceFingerprint(config)
        
        # Create a test image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img = Image.new('RGB', (100, 100))
            img.save(f.name)
            temp_image = f.name
        
        try:
            invalid_fingerprint = {'invalid': 'format'}
            with self.assertRaises(Exception):
                engine.verify_image(temp_image, invalid_fingerprint)
        finally:
            os.unlink(temp_image)

if __name__ == '__main__':
    unittest.main(verbosity=2)
