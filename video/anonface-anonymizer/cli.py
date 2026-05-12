"""Command-line interface for face anonymization."""

import argparse
import cv2
import numpy as np
import os
import shutil
import sys
from anonymizer import FaceAnonymizer


def get_image_files(directory: str, recursive: bool = True) -> list:
    """Get image files from directory (optionally recursive)."""
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif', '.gif')
    files = []
    if recursive:
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if filename.lower().endswith(extensions):
                    files.append(os.path.join(root, filename))
    else:
        for filename in os.listdir(directory):
            if filename.lower().endswith(extensions):
                files.append(os.path.join(directory, filename))
    return sorted(files)


def process_image(input_path: str, output_path: str, anonymizer: FaceAnonymizer, mode: str) -> bool:
    """Process single image."""
    try:
        # Windows'ta Türkçe karakter sorununu çözmek için numpy ile oku
        img_array = np.fromfile(input_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARNING] Failed to load: {input_path}", file=sys.stderr)
            return False
        
        anonymized = anonymizer.anonymize(img, mode)
        if anonymized is None:
            print(f"[WARNING] No face detected: {input_path}", file=sys.stderr)
            return False
        
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Windows'ta Türkçe karakter sorununu çözmek için encode/decode kullan
        ext = os.path.splitext(output_path)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            encode_ext = '.jpg'
        elif ext == '.png':
            encode_ext = '.png'
        elif ext == '.bmp':
            encode_ext = '.bmp'
        elif ext == '.webp':
            encode_ext = '.webp'
        else:
            # Varsayılan olarak .jpg kullan
            encode_ext = '.jpg'
        
        # Encode ve dosyaya yaz
        ret, encoded_img = cv2.imencode(encode_ext, anonymized)
        if not ret:
            print(f"[WARNING] Failed to encode image: {output_path}", file=sys.stderr)
            return False
        
        encoded_img.tofile(output_path)
        print(f"[OK] Processed: {os.path.basename(input_path)}")
        return True
    except Exception as e:
        print(f"[ERROR] Error processing {input_path}: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Face anonymization tool - eyes and mouth only",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", "-i", required=True, 
                       help="Input image file or directory")
    parser.add_argument("--output", "-o", required=True,
                       help="Output image file or directory")
    from core.modes import get_available_modes
    available_modes = get_available_modes()
    parser.add_argument("--mode", "-m", choices=available_modes,
                       default="blur", help=f"Censoring mode: {', '.join(available_modes)} (default: blur)")
    parser.add_argument("--model", default="face_landmarker.task",
                       help="Path to face_landmarker.task model file")
    parser.add_argument(
        "--copy-on-fail",
        action="store_true",
        help="If processing fails, copy original image to output path"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"[ERROR] Input path does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    try:
        anonymizer = FaceAnonymizer(args.model)
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}", file=sys.stderr)
        sys.exit(1)
    
    if os.path.isfile(args.input):
        # Eğer output path'inde uzantı yoksa, input dosyasının uzantısını ekle
        output_path = args.output
        output_ext = os.path.splitext(output_path)[1].lower()
        if not output_ext:
            # Output'ta uzantı yok, input dosyasının uzantısını kullan
            input_ext = os.path.splitext(args.input)[1]
            if input_ext:
                output_path = output_path + input_ext
        
        success = process_image(args.input, output_path, anonymizer, args.mode)
        if not success:
            sys.exit(1)
    
    elif os.path.isdir(args.input):
        image_files = get_image_files(args.input, recursive=True)
        
        if not image_files:
            print(f"[WARNING] No valid images found in directory: {args.input}", file=sys.stderr)
            sys.exit(1)
        
        os.makedirs(args.output, exist_ok=True)
        
        success_count = 0
        fail_count = 0
        
        for img_path in image_files:
            # Keep relative path and file name under output root.
            rel_path = os.path.relpath(img_path, args.input)
            output_path = os.path.join(args.output, rel_path)
            
            if process_image(img_path, output_path, anonymizer, args.mode):
                success_count += 1
            else:
                if args.copy_on_fail:
                    try:
                        output_dir = os.path.dirname(output_path)
                        if output_dir:
                            os.makedirs(output_dir, exist_ok=True)
                        shutil.copy2(img_path, output_path)
                        print(f"[FALLBACK] Copied original: {os.path.basename(img_path)}")
                        success_count += 1
                        continue
                    except Exception as copy_err:
                        print(f"[ERROR] Fallback copy failed for {img_path}: {copy_err}", file=sys.stderr)
                fail_count += 1
        
        print(f"\n[OK] Processed {success_count} images, {fail_count} failed")
        if fail_count > 0:
            sys.exit(1)
    
    else:
        print(f"[ERROR] Invalid input path: {args.input}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

