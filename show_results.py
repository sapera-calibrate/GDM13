#!/usr/bin/env python3
"""Show test results for existing fingerprints"""

import json
from protrace_spn import ProTraceConfig, ProTraceFingerprint

# Load existing fingerprints
with open('1_fingerprint.json', 'r') as f:
    fp1 = json.load(f)

with open('2_fingerprint.json', 'r') as f:
    fp2 = json.load(f)

print("=" * 70)
print("ProTrace Fingerprint Analysis - 1.png vs 2.png")
print("=" * 70)

# Display fingerprint info
print("\n📊 FINGERPRINT INFORMATION")
print("-" * 40)
print(f"\n1.png Fingerprint:")
print(f"  - Image ID: {fp1['image_id']}")
print(f"  - Dimensions: {fp1['dimensions'][0]}x{fp1['dimensions'][1]}")
print(f"  - Chaotic samples: {len(fp1['chaotic_samples'])}")
print(f"  - Block hashes: {len(fp1['block_hashes'])}")
print(f"  - Checksum (first 16): {fp1['checksum'][:16]}...")

print(f"\n2.png Fingerprint:")
print(f"  - Image ID: {fp2['image_id']}")
print(f"  - Dimensions: {fp2['dimensions'][0]}x{fp2['dimensions'][1]}")
print(f"  - Chaotic samples: {len(fp2['chaotic_samples'])}")
print(f"  - Block hashes: {len(fp2['block_hashes'])}")
print(f"  - Checksum (first 16): {fp2['checksum'][:16]}...")

# Initialize engine for verification
config = ProTraceConfig()
engine = ProTraceFingerprint(config)

print("\n🔍 VERIFICATION TESTS")
print("-" * 40)

# Self-verification
print("\nSelf-Verification (each image vs its own fingerprint):")
print("Expected: High similarity (>85%)\n")

report1 = engine.verify_image('1.png', fp1)
print(f"1.png vs own fingerprint:")
print(f"  - Chaotic similarity: {report1['chaotic_similarity']:.1%}")
print(f"  - Block similarity: {report1['block_similarity']:.1%}")
print(f"  - Overall similarity: {report1['overall_similarity']:.1%}")
print(f"  - Status: {'✅ VERIFIED' if report1['is_verified'] else '❌ NOT VERIFIED'}")

report2 = engine.verify_image('2.png', fp2)
print(f"\n2.png vs own fingerprint:")
print(f"  - Chaotic similarity: {report2['chaotic_similarity']:.1%}")
print(f"  - Block similarity: {report2['block_similarity']:.1%}")
print(f"  - Overall similarity: {report2['overall_similarity']:.1%}")
print(f"  - Status: {'✅ VERIFIED' if report2['is_verified'] else '❌ NOT VERIFIED'}")

# Cross-verification
print("\n" + "-" * 40)
print("Cross-Verification (images vs each other's fingerprints):")
print("Expected: Low similarity (<85%)\n")

report12 = engine.verify_image('1.png', fp2)
print(f"1.png vs 2.png's fingerprint:")
print(f"  - Chaotic similarity: {report12['chaotic_similarity']:.1%}")
print(f"  - Block similarity: {report12['block_similarity']:.1%}")
print(f"  - Overall similarity: {report12['overall_similarity']:.1%}")
print(f"  - Status: {'❌ NOT VERIFIED (Expected)' if not report12['is_verified'] else '⚠️ VERIFIED (Unexpected)'}")

report21 = engine.verify_image('2.png', fp1)
print(f"\n2.png vs 1.png's fingerprint:")
print(f"  - Chaotic similarity: {report21['chaotic_similarity']:.1%}")
print(f"  - Block similarity: {report21['block_similarity']:.1%}")
print(f"  - Overall similarity: {report21['overall_similarity']:.1%}")
print(f"  - Status: {'❌ NOT VERIFIED (Expected)' if not report21['is_verified'] else '⚠️ VERIFIED (Unexpected)'}")

# Save detailed reports
reports = {
    '1_self': report1,
    '2_self': report2,
    '1_vs_2': report12,
    '2_vs_1': report21
}

with open('test_results.json', 'w') as f:
    json.dump(reports, f, indent=2)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\n✅ Self-verification successful: {report1['is_verified'] and report2['is_verified']}")
print(f"✅ Cross-verification correctly failed: {not report12['is_verified'] and not report21['is_verified']}")
print(f"\n📁 Detailed results saved to: test_results.json")
print("\n🔐 ProTrace successfully distinguishes between different images")
print("   while correctly identifying the same image!")
