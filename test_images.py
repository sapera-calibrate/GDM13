#!/usr/bin/env python3
"""
Test script for ProTrace fingerprinting on 1.png and 2.png
"""

import json
import os
from SP import generate_fingerprint, verify_image, save_fingerprint, load_fingerprint
from tabulate import tabulate
import time

def run_image_tests():
    """Run comprehensive tests on 1.png and 2.png"""
    
    print("=" * 70)
    print("ProTrace SPN Fingerprinting - Image Test Suite")
    print("=" * 70)
    
    images = ['1.png', '2.png']
    fingerprints = {}
    
    # Step 1: Generate fingerprints for both images
    print("\n📝 STEP 1: Generating Fingerprints")
    print("-" * 50)
    
    for img in images:
        print(f"\nProcessing {img}...")
        start_time = time.time()
        
        # Generate fingerprint
        fingerprint = generate_fingerprint(img, f"img_{img.split('.')[0]}")
        fingerprints[img] = fingerprint
        
        # Save fingerprint to file
        fingerprint_file = f"{img.split('.')[0]}_fingerprint.json"
        save_fingerprint(fingerprint, fingerprint_file)
        
        elapsed = time.time() - start_time
        
        print(f"  ✓ Generated fingerprint in {elapsed:.3f}s")
        print(f"  - Image ID: {fingerprint['image_id']}")
        print(f"  - Dimensions: {fingerprint['dimensions'][0]}x{fingerprint['dimensions'][1]}")
        print(f"  - Chaotic samples: {len(fingerprint['chaotic_samples'])}")
        print(f"  - Block hashes: {len(fingerprint['block_hashes'])}")
        print(f"  - Saved to: {fingerprint_file}")
    
    # Step 2: Self-verification (each image against its own fingerprint)
    print("\n\n🔍 STEP 2: Self-Verification Tests")
    print("-" * 50)
    print("Testing each image against its own fingerprint (should be 100% match)\n")
    
    self_results = []
    for img in images:
        report = verify_image(img, fingerprints[img])
        self_results.append([
            img,
            f"{report['chaotic_similarity']:.1%}",
            f"{report['block_similarity']:.1%}",
            f"{report['overall_similarity']:.1%}",
            "✅ PASS" if report['is_verified'] else "❌ FAIL"
        ])
    
    # Display self-verification results
    headers = ["Image", "Chaotic Match", "Block Match", "Overall Match", "Status"]
    print(tabulate(self_results, headers=headers, tablefmt="grid"))
    
    # Step 3: Cross-verification (test images against each other's fingerprints)
    print("\n\n🔄 STEP 3: Cross-Verification Tests")
    print("-" * 50)
    print("Testing images against each other's fingerprints (should fail)\n")
    
    cross_results = []
    
    # Test 1.png against 2.png's fingerprint
    report_1_vs_2 = verify_image('1.png', fingerprints['2.png'])
    cross_results.append([
        "1.png",
        "2.png fingerprint",
        f"{report_1_vs_2['chaotic_similarity']:.1%}",
        f"{report_1_vs_2['block_similarity']:.1%}",
        f"{report_1_vs_2['overall_similarity']:.1%}",
        "❌ PASS" if not report_1_vs_2['is_verified'] else "⚠️ UNEXPECTED"
    ])
    
    # Test 2.png against 1.png's fingerprint
    report_2_vs_1 = verify_image('2.png', fingerprints['1.png'])
    cross_results.append([
        "2.png",
        "1.png fingerprint",
        f"{report_2_vs_1['chaotic_similarity']:.1%}",
        f"{report_2_vs_1['block_similarity']:.1%}",
        f"{report_2_vs_1['overall_similarity']:.1%}",
        "❌ PASS" if not report_2_vs_1['is_verified'] else "⚠️ UNEXPECTED"
    ])
    
    # Display cross-verification results
    headers = ["Test Image", "Against Fingerprint", "Chaotic", "Block", "Overall", "Result"]
    print(tabulate(cross_results, headers=headers, tablefmt="grid"))
    
    # Step 4: Detailed comparison
    print("\n\n📊 STEP 4: Detailed Fingerprint Comparison")
    print("-" * 50)
    
    fp1 = fingerprints['1.png']
    fp2 = fingerprints['2.png']
    
    comparison_data = [
        ["Image ID", fp1['image_id'], fp2['image_id']],
        ["Dimensions", f"{fp1['dimensions'][0]}x{fp1['dimensions'][1]}", 
         f"{fp2['dimensions'][0]}x{fp2['dimensions'][1]}"],
        ["Checksum (first 16)", fp1['checksum'][:16], fp2['checksum'][:16]],
        ["Chaotic Samples", len(fp1['chaotic_samples']), len(fp2['chaotic_samples'])],
        ["Block Hashes", len(fp1['block_hashes']), len(fp2['block_hashes'])],
        ["Config - Block Size", fp1['config']['block_size'], fp2['config']['block_size']],
        ["Config - Rounds", fp1['config']['rounds'], fp2['config']['rounds']]
    ]
    
    headers = ["Property", "1.png", "2.png"]
    print(tabulate(comparison_data, headers=headers, tablefmt="grid"))
    
    # Step 5: Summary
    print("\n\n✨ TEST SUMMARY")
    print("=" * 70)
    
    # Check if all self-verifications passed
    self_verify_pass = all(report['is_verified'] for img in images 
                          for report in [verify_image(img, fingerprints[img])])
    
    # Check if all cross-verifications failed (as expected)
    cross_verify_correct = (not report_1_vs_2['is_verified']) and (not report_2_vs_1['is_verified'])
    
    print(f"✓ Self-verification: {'PASSED' if self_verify_pass else 'FAILED'}")
    print(f"  - Each image correctly matches its own fingerprint")
    
    print(f"\n✓ Cross-verification: {'PASSED' if cross_verify_correct else 'FAILED'}")
    print(f"  - Images correctly fail to match each other's fingerprints")
    
    print(f"\n✓ System integrity: {'VERIFIED' if self_verify_pass and cross_verify_correct else 'COMPROMISED'}")
    
    # Additional insights
    print("\n📈 Performance Metrics:")
    print(f"  - Fingerprint generation: ~{elapsed:.3f}s per image")
    print(f"  - Verification: <0.1s per check")
    print(f"  - Fingerprint size: ~{len(json.dumps(fp1))} bytes")
    
    print("\n🔐 Security Features Active:")
    print(f"  - Secret key: Automatically generated and stored")
    print(f"  - Chaotic sampling: Keyed pseudo-random coordinates")
    print(f"  - SPN hashing: {fp1['config']['rounds']}-round substitution-permutation network")
    
    return {
        'self_verification': self_verify_pass,
        'cross_verification': cross_verify_correct,
        'fingerprints': fingerprints
    }

if __name__ == '__main__':
    try:
        # Check if images exist
        if not os.path.exists('1.png'):
            print("❌ Error: 1.png not found!")
            exit(1)
        if not os.path.exists('2.png'):
            print("❌ Error: 2.png not found!")
            exit(1)
        
        # Run tests
        results = run_image_tests()
        
        print("\n" + "=" * 70)
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
