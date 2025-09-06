#!/usr/bin/env python3
"""
SP.py - Main entry point for ProTrace SPN fingerprinting system
Production-ready image fingerprinting with keyed chaotic coordinates, 
block SPN hashes, and compact checksums.
"""

from protrace_spn import (
    ProTraceConfig,
    ProTraceFingerprint,
    ProTraceCLI,
    VERSION
)
import json
import logging

# Export main components for direct import
__all__ = [
    'ProTraceConfig',
    'ProTraceFingerprint', 
    'ProTraceCLI',
    'VERSION',
    'generate_fingerprint',
    'verify_image',
    'load_fingerprint',
    'save_fingerprint'
]

# Configure module logger
logger = logging.getLogger(__name__)

def generate_fingerprint(image_path: str, image_id: str, config: ProTraceConfig = None):
    """
    Convenience function to generate a fingerprint
    
    Args:
        image_path: Path to the image file
        image_id: Unique identifier for the image
        config: Optional ProTraceConfig object
    
    Returns:
        Dictionary containing the fingerprint data
    
    Example:
        >>> fingerprint = generate_fingerprint('image.png', 'img_001')
        >>> save_fingerprint(fingerprint, 'fingerprint.json')
    """
    if config is None:
        config = ProTraceConfig()
    
    engine = ProTraceFingerprint(config)
    return engine.generate_fingerprint(image_path, image_id)

def verify_image(image_path: str, fingerprint: dict, config: ProTraceConfig = None):
    """
    Convenience function to verify an image against a fingerprint
    
    Args:
        image_path: Path to the candidate image
        fingerprint: Fingerprint dictionary to verify against
        config: Optional ProTraceConfig object
    
    Returns:
        Dictionary containing the verification report with:
        - is_verified: Boolean indicating if verification passed
        - overall_similarity: Float similarity score (0-1)
        - chaotic_similarity: Float score for chaotic point matches
        - block_similarity: Float score for block hash matches
    
    Example:
        >>> fingerprint = load_fingerprint('fingerprint.json')
        >>> report = verify_image('candidate.png', fingerprint)
        >>> if report['is_verified']:
        ...     print(f"Image verified: {report['overall_similarity']:.1%}")
    """
    if config is None:
        config = ProTraceConfig()
        # Use config from fingerprint if available
        if 'config' in fingerprint:
            config.block_size = fingerprint['config'].get('block_size', config.block_size)
            config.num_chaotic_points = fingerprint['config'].get('num_chaotic_points', config.num_chaotic_points)
            config.rounds = fingerprint['config'].get('rounds', config.rounds)
    
    engine = ProTraceFingerprint(config)
    return engine.verify_image(image_path, fingerprint)

def load_fingerprint(filepath: str) -> dict:
    """
    Load a fingerprint from a JSON file
    
    Args:
        filepath: Path to the fingerprint JSON file
    
    Returns:
        Fingerprint dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)

def save_fingerprint(fingerprint: dict, filepath: str):
    """
    Save a fingerprint to a JSON file
    
    Args:
        fingerprint: Fingerprint dictionary to save
        filepath: Path where to save the fingerprint
    """
    with open(filepath, 'w') as f:
        json.dump(fingerprint, f, indent=2)
    logger.info(f"Fingerprint saved to {filepath}")

if __name__ == '__main__':
    # Run CLI when executed directly
    import sys
    cli = ProTraceCLI()
    sys.exit(cli.run())


