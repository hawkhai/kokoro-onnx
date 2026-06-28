"""
Convert voices file from numpy format to Android-compatible format.

Usage:
    python convert_voices.py input.bin output.npz

This script converts the voices-v1.0.bin file (which is a numpy .npz file)
to ensure compatibility with the Android NumpyLoader.

The voices file contains multiple voice style arrays, each indexed by
token length. Each voice has shape (max_token_length, style_dim).
"""

import sys
import numpy as np


def convert_voices(input_path: str, output_path: str):
    """Convert voices file to Android-compatible format."""
    print(f"Loading voices from: {input_path}")

    # Load voices (numpy .npz or .npy format)
    try:
        voices = np.load(input_path)
    except Exception as e:
        print(f"Error loading voices: {e}")
        print("Make sure the file is a valid numpy .npy or .npz file")
        sys.exit(1)

    print(f"Loaded {len(voices)} voices:")
    for name in sorted(voices.keys()):
        data = voices[name]
        print(f"  {name}: shape={data.shape}, dtype={data.dtype}")

    # Save in .npz format (compatible with Android NumpyLoader)
    print(f"\nSaving to: {output_path}")
    np.savez(output_path, **{name: voices[name] for name in voices.keys()})

    print("Conversion complete!")
    print(f"\nPlace '{output_path}' in your Android assets folder.")


def main():
    if len(sys.argv) != 3:
        print("Usage: python convert_voices.py <input.bin> <output.npz>")
        print("\nExample:")
        print("  python convert_voices.py voices-v1.0.bin voices_android.npz")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    convert_voices(input_path, output_path)


if __name__ == "__main__":
    main()
