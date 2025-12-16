#!/usr/bin/env python3
"""
OCR Script for RAG Pipeline - Text Output
Converts scanned documents (PDF or images) into plain text files.

Usage:
    python scripts/ocr_to_text.py input/your_file.pdf
    python scripts/ocr_to_text.py input/your_file.png --lang eng+heb
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict

try:
    import pytesseract
    from PIL import Image
    from pdf2image import convert_from_path
except ImportError as e:
    print(f"Error: Missing required package - {e}")
    print("\nPlease install OCR dependencies:")
    print("pip install pytesseract pdf2image pillow")
    print("\nYou also need to install Tesseract OCR:")
    print("- macOS: brew install tesseract tesseract-lang")
    print("- Ubuntu: sudo apt-get install tesseract-ocr tesseract-ocr-heb")
    print("- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
    sys.exit(1)


# Configuration
DEFAULT_CONFIG = {
    "language": "eng",  # Default to English. Can be: eng, heb, eng+heb, etc.
}


def detect_type(input_path: str) -> str:
    """
    Detect if input file is a PDF or an image.

    Args:
        input_path: Path to input file

    Returns:
        'pdf' or 'image'

    Raises:
        ValueError: If file type is not supported
    """
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = path.suffix.lower()

    if suffix == '.pdf':
        return 'pdf'
    elif suffix in ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']:
        return 'image'
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: .pdf, .png, .jpg, .jpeg, .tiff, .bmp")


def images_from_input(input_path: str) -> List[Image.Image]:
    """
    Convert input file to a list of PIL Images.
    - If PDF: convert each page to an image
    - If image: load it directly

    Args:
        input_path: Path to input file

    Returns:
        List of PIL Image objects
    """
    file_type = detect_type(input_path)

    if file_type == 'pdf':
        print(f"Converting PDF pages to images...")
        # Convert PDF pages to images
        # dpi=300 for good quality OCR
        images = convert_from_path(input_path, dpi=300)
        print(f"Converted {len(images)} pages")
        return images
    else:
        # Load image file directly
        print(f"Loading image file...")
        image = Image.open(input_path)
        return [image]


def ocr_image_to_text(image: Image.Image, lang: str) -> str:
    """
    Run OCR on an image and extract text.

    Args:
        image: PIL Image object
        lang: Tesseract language code (e.g., 'eng', 'heb', 'eng+heb')

    Returns:
        Extracted text as string
    """
    # Run OCR to extract text
    text = pytesseract.image_to_string(
        image,
        lang=lang,
        config='--psm 1'  # Automatic page segmentation with OSD
    )

    return text


def create_ocr_text(input_path: str, output_text_path: str, config: Dict[str, str]) -> str:
    """
    Main OCR function: Convert a scanned document into plain text.

    Args:
        input_path: Path to input file (PDF or image)
        output_text_path: Path where to save the text file
        config: Configuration dictionary with 'language' key

    Returns:
        Path to the output text file
    """
    lang = config.get('language', 'eng')

    print(f"\nStarting OCR process...")
    print(f"Input: {input_path}")
    print(f"Output: {output_text_path}")
    print(f"Language: {lang}")
    print("-" * 50)

    # Step 1: Convert input to images
    images = images_from_input(input_path)

    # Step 2: Run OCR on each image and extract text
    print(f"\nRunning OCR on {len(images)} image(s)...")
    all_text = []

    for i, image in enumerate(images, 1):
        print(f"Processing page {i}/{len(images)}...")
        text = ocr_image_to_text(image, lang)

        # Add page separator for multi-page documents
        if len(images) > 1:
            all_text.append(f"{'='*60}")
            all_text.append(f"Page {i}")
            all_text.append(f"{'='*60}")

        all_text.append(text)

        # Add spacing between pages
        if i < len(images):
            all_text.append("\n\n")

    # Step 3: Combine all text and save to file
    print(f"\nSaving text to file...")
    combined_text = "\n".join(all_text)

    with open(output_text_path, 'w', encoding='utf-8') as f:
        f.write(combined_text)

    print(f"\n✓ OCR complete!")
    print(f"Text file saved to: {output_text_path}")
    print(f"Total characters: {len(combined_text)}")

    return output_text_path


def generate_output_path(input_path: str, output_dir: str = "data") -> str:
    """
    Generate output text file path by adding _ocr suffix.

    Args:
        input_path: Original input file path
        output_dir: Directory where to save output (default: 'data')

    Returns:
        Output text file path

    Examples:
        invoice.pdf -> data/invoice_ocr.txt
        scan_01.png -> data/scan_01_ocr.txt
    """
    input_file = Path(input_path)
    stem = input_file.stem  # filename without extension

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Add _ocr suffix and .txt extension
    output_filename = f"{stem}_ocr.txt"
    return str(output_path / output_filename)


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Convert scanned documents to plain text using OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/ocr_to_text.py input/invoice.pdf
  python scripts/ocr_to_text.py input/scan.png --lang heb
  python scripts/ocr_to_text.py input/doc.pdf --lang eng+heb --output custom_dir
        """
    )

    parser.add_argument(
        'input_path',
        help='Path to input file (PDF or image)'
    )

    parser.add_argument(
        '--lang',
        default=DEFAULT_CONFIG['language'],
        help=f"Tesseract language code (default: {DEFAULT_CONFIG['language']}). "
             "Examples: eng, heb, eng+heb, ara, fra"
    )

    parser.add_argument(
        '--output',
        default='data',
        help='Output directory (default: data)'
    )

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input_path):
        print(f"Error: Input file not found: {args.input_path}")
        sys.exit(1)

    # Generate output path
    output_text_path = generate_output_path(args.input_path, args.output)

    # Prepare config
    config = {
        'language': args.lang
    }

    # Run OCR
    try:
        result_path = create_ocr_text(args.input_path, output_text_path, config)
        print(f"\n{'=' * 50}")
        print(f"SUCCESS: {result_path}")
        print(f"{'=' * 50}")
        return 0
    except Exception as e:
        print(f"\nError during OCR processing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
