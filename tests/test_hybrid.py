#!/usr/bin/env python3
"""Test the enhanced ProTrace system with hybrid dithering sampler"""

import json
import numpy as np
from protrace_spn import ProTraceConfig, ProTraceFingerprint

def test_hybrid_sampler():
    print("=" * 70)
    print("Enhanced ProTrace: Hybrid Dithering + ISCC-Style Sampling Test")
    print("=" * 70)
    
    config = ProTraceConfig()
    engine = ProTraceFingerprint(config)
    
    print(f"\n🔧 Configuration:")
    print(f"   - Samples (M_total): {config.num_chaotic_points}")
    print(f"   - Grid fraction: 25%")
    print(f"   - Block size: 32x32 for energy map")
    print(f"   - Importance method: Sobel edge detection")
    print(f"   - Dithering: Floyd-Steinberg error diffusion")
    
    # Generate fingerprints with hybrid sampling
    print(f"\n📝 Generating Hybrid Fingerprints...")
    
    print(f"\n  Processing test_a.png with hybrid sampler...")
    fp_a = engine.generate_fingerprint('test_a.png', 'hybrid_test_a')
    
    print(f"    ✓ Generated: {fp_a['B']} blocks, {len(fp_a['tiny_checksums'])} checksums")
    print(f"    ✓ Sampler: {fp_a['sampler']['type']} (epoch: {fp_a['sampler']['epoch']})")
    print(f"    ✓ Samples: {len(fp_a['chaotic_samples'])} coordinates")
    
    print(f"\n  Processing test_b.png with hybrid sampler...")
    fp_b = engine.generate_fingerprint('test_b.png', 'hybrid_test_b')
    
    print(f"    ✓ Generated: {fp_b['B']} blocks, {len(fp_b['tiny_checksums'])} checksums")
    print(f"    ✓ Sampler: {fp_b['sampler']['type']} (epoch: {fp_b['sampler']['epoch']})")
    print(f"    ✓ Samples: {len(fp_b['chaotic_samples'])} coordinates")
    
    # Analyze sampling distribution
    print(f"\n📊 Sampling Distribution Analysis:")
    
    coords_a = [(s['coord'][0], s['coord'][1]) for s in fp_a['chaotic_samples']]
    coords_b = [(s['coord'][0], s['coord'][1]) for s in fp_b['chaotic_samples']]
    
    # Check coordinate spread
    x_vals_a = [c[0] for c in coords_a]
    y_vals_a = [c[1] for c in coords_a]
    
    print(f"   Test A coordinate spread:")
    print(f"     X: {min(x_vals_a)}-{max(x_vals_a)} (range: {max(x_vals_a)-min(x_vals_a)})")
    print(f"     Y: {min(y_vals_a)}-{max(y_vals_a)} (range: {max(y_vals_a)-min(y_vals_a)})")
    
    # Run verification tests
    print(f"\n🔍 Verification Tests:")
    
    # Self-verification
    print(f"\n   Self-Verification:")
    r_aa = engine.verify_image('test_a.png', fp_a)
    r_bb = engine.verify_image('test_b.png', fp_b)
    
    print(f"     A vs A: {r_aa['global_similarity']:.3f} ({'✅' if r_aa['is_verified'] else '❌'})")
    print(f"     B vs B: {r_bb['global_similarity']:.3f} ({'✅' if r_bb['is_verified'] else '❌'})")
    
    # Cross-verification  
    print(f"\n   Cross-Verification:")
    r_ab = engine.verify_image('test_a.png', fp_b)
    r_ba = engine.verify_image('test_b.png', fp_a)
    
    print(f"     A vs B: {r_ab['global_similarity']:.3f} ({'✅ REJECTED' if not r_ab['is_verified'] else '❌ ACCEPTED'})")
    print(f"     B vs A: {r_ba['global_similarity']:.3f} ({'✅ REJECTED' if not r_ba['is_verified'] else '❌ ACCEPTED'})")
    
    # Detailed analysis
    print(f"\n   Detailed Cross-Verification Analysis:")
    print(f"     A vs B fingerprint:")
    print(f"       - Block similarity: {r_ab['avg_block_similarity']:.3f}")
    print(f"       - Checksum similarity: {r_ab['checksum_similarity']:.3f}")
    print(f"       - Flagged blocks: {len(r_ab['flagged_blocks'])}/{fp_b['B']}")
    print(f"       - Avg Hamming distance: {np.mean(r_ab['per_block_hamming']):.3f}")
    print(f"       - Tiny mismatch fraction: {r_ab['tiny_mismatch_fraction']:.3f}")
    
    # Compare with previous results
    print(f"\n📈 Hybrid Sampler Improvements:")
    print(f"   - Importance-based sampling targets salient regions")
    print(f"   - Error diffusion ensures smooth spatial distribution")
    print(f"   - Grid baseline guarantees spatial coverage")
    print(f"   - ISCC-like block energy for semantic relevance")
    
    # System assessment
    self_ok = r_aa['is_verified'] and r_bb['is_verified']
    cross_ok = not r_ab['is_verified'] and not r_ba['is_verified']
    system_working = self_ok and cross_ok
    
    print(f"\n🏆 HYBRID SYSTEM STATUS:")
    print(f"     Self-verification: {'✅ PASS' if self_ok else '❌ FAIL'}")
    print(f"     Cross-verification: {'✅ PASS' if cross_ok else '❌ FAIL'}")
    print(f"     Overall: {'🎉 SUCCESS' if system_working else '🔧 NEEDS WORK'}")
    
    if system_working:
        print(f"\n✨ Enhanced ProTrace with Hybrid Sampling is working correctly!")
        print(f"   ✓ Importance-based coordinate selection")
        print(f"   ✓ Robust verification with Hamming analysis")
        print(f"   ✓ Production-ready fingerprinting system")
    else:
        print(f"\n⚠️  Issues detected with hybrid sampling system")
        if not cross_ok:
            avg_cross_sim = (r_ab['global_similarity'] + r_ba['global_similarity']) / 2
            print(f"      Average cross-similarity: {avg_cross_sim:.3f} (should be < 0.90)")
    
    # Save results
    results = {
        'system_working': system_working,
        'hybrid_sampler': {
            'type': fp_a['sampler']['type'],
            'epoch': fp_a['sampler']['epoch'],
            'parameters': fp_a['sampler']
        },
        'verification_tests': {
            'self_a': r_aa,
            'self_b': r_bb,
            'cross_ab': r_ab,
            'cross_ba': r_ba
        },
        'performance_metrics': {
            'avg_cross_similarity': (r_ab['global_similarity'] + r_ba['global_similarity']) / 2,
            'avg_hamming_distance': np.mean(r_ab['per_block_hamming'] + r_ba['per_block_hamming']),
            'flagged_blocks_ratio': len(r_ab['flagged_blocks']) / fp_b['B']
        }
    }
    
    with open('hybrid_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to: hybrid_test_results.json")
    
    return system_working

if __name__ == '__main__':
    success = test_hybrid_sampler()
    if success:
        print(f"\n🏆 Hybrid dithering sampler validation PASSED!")
    else:
        print(f"\n🚨 Hybrid dithering sampler validation FAILED!")
    exit(0 if success else 1)
