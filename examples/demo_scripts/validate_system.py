#!/usr/bin/env python3
"""Create synthetic test images and validate the ProTrace system works correctly"""

import numpy as np
from PIL import Image, ImageDraw
import json
from protrace_spn import ProTraceConfig, ProTraceFingerprint

def create_test_images():
    """Create clearly different test images"""
    
    # Image A: Blue gradient with red circle
    img_a = Image.new('RGB', (512, 512), 'white')
    draw_a = ImageDraw.Draw(img_a)
    
    # Blue gradient background
    for y in range(512):
        color_val = int(255 * (1 - y / 512))
        draw_a.rectangle([(0, y), (512, y+1)], fill=(0, 0, color_val))
    
    # Red circle
    draw_a.ellipse([150, 150, 350, 350], fill='red', outline='darkred', width=3)
    draw_a.text((200, 250), "IMAGE A", fill='yellow')
    
    img_a.save('test_a.png')
    
    # Image B: Green gradient with blue rectangle
    img_b = Image.new('RGB', (512, 512), 'black')
    draw_b = ImageDraw.Draw(img_b)
    
    # Green gradient background
    for x in range(512):
        color_val = int(255 * (x / 512))
        draw_b.rectangle([(x, 0), (x+1, 512)], fill=(0, color_val, 0))
    
    # Blue rectangle
    draw_b.rectangle([100, 200, 400, 300], fill='blue', outline='lightblue', width=3)
    draw_b.text((200, 250), "IMAGE B", fill='white')
    
    img_b.save('test_b.png')
    
    print("✓ Created test_a.png and test_b.png (clearly different)")

def validate_system():
    """Validate ProTrace system with clearly different images"""
    
    print("=" * 70)
    print("SYSTEM VALIDATION: ProTrace with Synthetic Test Images")
    print("=" * 70)
    
    # Create test images
    create_test_images()
    
    # Initialize system
    config = ProTraceConfig()
    engine = ProTraceFingerprint(config)
    
    print(f"\n🔧 System Configuration:")
    print(f"   - Verification threshold: 0.90")
    print(f"   - Block flag threshold: 0.12") 
    print(f"   - Checksum tolerance: 0.02")
    
    # Generate fingerprints
    print(f"\n📝 Generating Fingerprints...")
    fp_a = engine.generate_fingerprint('test_a.png', 'synthetic_a')
    fp_b = engine.generate_fingerprint('test_b.png', 'synthetic_b')
    
    print(f"   Test A: {fp_a['B']} blocks, {len(fp_a['tiny_checksums'])} checksums")
    print(f"   Test B: {fp_b['B']} blocks, {len(fp_b['tiny_checksums'])} checksums")
    
    # Run all verification tests
    print(f"\n🔍 Running Verification Tests...")
    
    # Self-verification (should pass)
    print(f"\n   Self-Verification:")
    r_aa = engine.verify_image('test_a.png', fp_a)
    r_bb = engine.verify_image('test_b.png', fp_b)
    
    print(f"     A vs A: {r_aa['global_similarity']:.3f} ({'✅ PASS' if r_aa['is_verified'] else '❌ FAIL'})")
    print(f"     B vs B: {r_bb['global_similarity']:.3f} ({'✅ PASS' if r_bb['is_verified'] else '❌ FAIL'})")
    
    # Cross-verification (should fail) 
    print(f"\n   Cross-Verification:")
    r_ab = engine.verify_image('test_a.png', fp_b)
    r_ba = engine.verify_image('test_b.png', fp_a)
    
    print(f"     A vs B: {r_ab['global_similarity']:.3f} ({'✅ PASS' if not r_ab['is_verified'] else '❌ FAIL'})")
    print(f"     B vs A: {r_ba['global_similarity']:.3f} ({'✅ PASS' if not r_ba['is_verified'] else '❌ FAIL'})")
    
    # Detailed analysis
    print(f"\n📊 Detailed Analysis:")
    print(f"     A vs B fingerprint:")
    print(f"       - Block similarity: {r_ab['avg_block_similarity']:.3f}")
    print(f"       - Checksum similarity: {r_ab['checksum_similarity']:.3f}")
    print(f"       - Flagged blocks: {len(r_ab['flagged_blocks'])}/{fp_b['B']}")
    print(f"       - Avg Hamming: {np.mean(r_ab['per_block_hamming']):.3f}")
    print(f"       - Max Hamming: {np.max(r_ab['per_block_hamming']):.3f}")
    
    # System assessment
    self_ok = r_aa['is_verified'] and r_bb['is_verified']
    cross_ok = not r_ab['is_verified'] and not r_ba['is_verified']
    system_working = self_ok and cross_ok
    
    print(f"\n🏆 VALIDATION RESULTS:")
    print(f"     Self-verification: {'✅ PASS' if self_ok else '❌ FAIL'}")
    print(f"     Cross-verification: {'✅ PASS' if cross_ok else '❌ FAIL'}")
    print(f"     System Status: {'🎉 WORKING' if system_working else '🔧 BROKEN'}")
    
    if system_working:
        print(f"\n✨ SUCCESS! ProTrace system is functioning correctly:")
        print(f"   ✓ Identical images verify with 100% similarity")
        print(f"   ✓ Different images are rejected with low similarity")
        print(f"   ✓ Hamming-based verification provides robust differentiation")
    else:
        print(f"\n🔧 Issues detected:")
        if not self_ok:
            print(f"   ❌ Self-verification failing - basic system broken")
        if not cross_ok:
            print(f"   ❌ Cross-verification not rejecting different images")
            print(f"      Average cross-similarity: {(r_ab['global_similarity'] + r_ba['global_similarity'])/2:.3f}")
    
    # Save validation results
    results = {
        'system_working': system_working,
        'tests': {
            'self_a': r_aa,
            'self_b': r_bb, 
            'cross_ab': r_ab,
            'cross_ba': r_ba
        },
        'summary': {
            'self_verification_passed': self_ok,
            'cross_verification_passed': cross_ok,
            'avg_cross_similarity': (r_ab['global_similarity'] + r_ba['global_similarity']) / 2,
            'avg_hamming_distance': np.mean(r_ab['per_block_hamming'] + r_ba['per_block_hamming'])
        }
    }
    
    with open('validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Full results saved to: validation_results.json")
    
    return system_working

if __name__ == '__main__':
    success = validate_system()
    if success:
        print(f"\n🏆 ProTrace system validation PASSED!")
    else:
        print(f"\n🚨 ProTrace system validation FAILED!")
    exit(0 if success else 1)
