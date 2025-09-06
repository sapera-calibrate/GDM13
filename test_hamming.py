#!/usr/bin/env python3
"""Test the updated Hamming-based ProTrace system"""

import json
import os
from protrace_spn import ProTraceConfig, ProTraceFingerprint

def test_hamming_system():
    print("=" * 70)
    print("Testing Updated ProTrace with Hamming-based Verification")
    print("=" * 70)
    
    # Initialize with updated config
    config = ProTraceConfig()
    engine = ProTraceFingerprint(config)
    
    print("\n🔧 System Configuration:")
    print(f"  - Verified threshold: 0.90")
    print(f"  - Per-block flag threshold: 0.12")
    print(f"  - Checksum mismatch tolerance: 0.02")
    print(f"  - Block weight: 0.8, Checksum weight: 0.2")
    
    # Generate new fingerprints with updated format
    print("\n📝 Generating Updated Fingerprints...")
    
    print("\n  Processing 1.png...")
    fp1 = engine.generate_fingerprint('1.png', 'img1_hamming')
    with open('1_fingerprint_hamming.json', 'w') as f:
        json.dump(fp1, f, indent=2)
    
    print(f"    ✓ Fingerprint generated")
    print(f"    - Dimensions: {fp1['dimensions']}")
    print(f"    - Block hashes (B): {fp1['B']}")
    print(f"    - Tiny checksums: {len(fp1['tiny_checksums'])}")
    print(f"    - Chaotic samples: {len(fp1['chaotic_samples'])}")
    
    print("\n  Processing 2.png...")
    fp2 = engine.generate_fingerprint('2.png', 'img2_hamming')
    with open('2_fingerprint_hamming.json', 'w') as f:
        json.dump(fp2, f, indent=2)
    
    print(f"    ✓ Fingerprint generated")
    print(f"    - Dimensions: {fp2['dimensions']}")
    print(f"    - Block hashes (B): {fp2['B']}")
    print(f"    - Tiny checksums: {len(fp2['tiny_checksums'])}")
    print(f"    - Chaotic samples: {len(fp2['chaotic_samples'])}")
    
    print("\n🔍 Running Verification Tests...")
    
    # Self-verification tests
    print("\n  Self-Verification Tests:")
    
    print("\n    1.png vs its own fingerprint:")
    report1 = engine.verify_image('1.png', fp1)
    print(f"      - Global similarity: {report1['global_similarity']:.3f}")
    print(f"      - Block similarity: {report1['avg_block_similarity']:.3f}")
    print(f"      - Checksum similarity: {report1['checksum_similarity']:.3f}")
    print(f"      - Flagged blocks: {len(report1['flagged_blocks'])}")
    print(f"      - Tiny mismatch fraction: {report1['tiny_mismatch_fraction']:.3f}")
    print(f"      - Status: {'✅ VERIFIED' if report1['is_verified'] else '❌ NOT VERIFIED'}")
    
    print("\n    2.png vs its own fingerprint:")
    report2 = engine.verify_image('2.png', fp2)
    print(f"      - Global similarity: {report2['global_similarity']:.3f}")
    print(f"      - Block similarity: {report2['avg_block_similarity']:.3f}")
    print(f"      - Checksum similarity: {report2['checksum_similarity']:.3f}")
    print(f"      - Flagged blocks: {len(report2['flagged_blocks'])}")
    print(f"      - Tiny mismatch fraction: {report2['tiny_mismatch_fraction']:.3f}")
    print(f"      - Status: {'✅ VERIFIED' if report2['is_verified'] else '❌ NOT VERIFIED'}")
    
    # Cross-verification tests
    print("\n  Cross-Verification Tests:")
    
    print("\n    1.png vs 2.png's fingerprint:")
    report12 = engine.verify_image('1.png', fp2)
    print(f"      - Global similarity: {report12['global_similarity']:.3f}")
    print(f"      - Block similarity: {report12['avg_block_similarity']:.3f}")
    print(f"      - Checksum similarity: {report12['checksum_similarity']:.3f}")
    print(f"      - Flagged blocks: {len(report12['flagged_blocks'])}")
    print(f"      - Tiny mismatch fraction: {report12['tiny_mismatch_fraction']:.3f}")
    print(f"      - Status: {'❌ NOT VERIFIED (Expected)' if not report12['is_verified'] else '⚠️ VERIFIED (Unexpected)'}")
    
    print("\n    2.png vs 1.png's fingerprint:")
    report21 = engine.verify_image('2.png', fp1)
    print(f"      - Global similarity: {report21['global_similarity']:.3f}")
    print(f"      - Block similarity: {report21['avg_block_similarity']:.3f}")
    print(f"      - Checksum similarity: {report21['checksum_similarity']:.3f}")
    print(f"      - Flagged blocks: {len(report21['flagged_blocks'])}")
    print(f"      - Tiny mismatch fraction: {report21['tiny_mismatch_fraction']:.3f}")
    print(f"      - Status: {'❌ NOT VERIFIED (Expected)' if not report21['is_verified'] else '⚠️ VERIFIED (Unexpected)'}")
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    self_verify_ok = report1['is_verified'] and report2['is_verified']
    cross_verify_ok = not report12['is_verified'] and not report21['is_verified']
    
    print(f"\n✅ Self-verification: {'PASSED' if self_verify_ok else 'FAILED'}")
    print(f"✅ Cross-verification: {'PASSED' if cross_verify_ok else 'FAILED'}")
    
    if cross_verify_ok:
        print("\n🎉 SUCCESS! The Hamming-based system correctly distinguishes between images.")
        print("   Cross-verification properly failed with low similarity scores.")
    else:
        print("\n⚠️ Cross-verification still shows high similarity.")
        print("   This suggests the images may be very similar or need stricter thresholds.")
    
    print(f"\n📊 Key Improvements:")
    print(f"   - Per-block Hamming distance analysis")
    print(f"   - Tiny checksum validation ({len(fp1['tiny_checksums'])} samples)")
    print(f"   - Conservative verification thresholds")
    print(f"   - Flagged blocks detection for localized changes")
    
    # Save detailed results
    results = {
        'self_verification': {
            '1_vs_1': report1,
            '2_vs_2': report2
        },
        'cross_verification': {
            '1_vs_2': report12,
            '2_vs_1': report21
        },
        'summary': {
            'self_verify_passed': self_verify_ok,
            'cross_verify_passed': cross_verify_ok,
            'system_working': self_verify_ok and cross_verify_ok
        }
    }
    
    with open('hamming_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Detailed results saved to: hamming_test_results.json")
    
    return self_verify_ok and cross_verify_ok

if __name__ == '__main__':
    success = test_hamming_system()
    if success:
        print("\n🏆 All tests passed! ProTrace system is working correctly.")
    else:
        print("\n🔧 Some tests failed. System may need further tuning.")
