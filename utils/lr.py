import os
from PIL import Image

def downsample_images(input_dir: str, output_dir: str, scale: float = 0.25):
    """
    Shrink all images in input_dir folder by scale ratio using bicubic interpolation
    and save them in output_dir folder.

    :param input_dir: Directory path containing original images
    :param output_dir: Directory path to save shrunk images
    :param scale: Reduction ratio (default 0.25 -> 1/4 size)
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Supported image extensions
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(exts):
            continue

        src_path = os.path.join(input_dir, fname)
        dst_path = os.path.join(output_dir, fname)

        try:
            with Image.open(src_path) as img:
                # Calculate new size
                new_w = int(img.width * scale)
                new_h = int(img.height * scale)

                # Resize (bicubic)
                img_resized = img.resize((new_w, new_h), resample=Image.BICUBIC)

                # Save
                img_resized.save(dst_path)
                print(f"Saved: {dst_path} ({new_w}×{new_h})")
        except Exception as e:
            print(f"Error processing {src_path}: {e}")

if __name__ == "__main__":
    # Example usage
    input_folder  = "../data/DIV2K_valid_HR"
    output_folder = "../data/no_use/DIV2K_valid_HR"
    downsample_images(input_folder, output_folder, scale=0.25)
