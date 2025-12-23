import os
from PIL import Image

def downsample_images(input_dir: str, output_dir: str, scale: float = 0.25):
    """
    input_dir 폴더의 모든 이미지를 bicubic 보간으로 scale 비율만큼 축소하여
    output_dir 폴더에 저장합니다.

    :param input_dir: 원본 이미지가 들어있는 디렉토리 경로
    :param output_dir: 축소된 이미지를 저장할 디렉토리 경로
    :param scale: 축소 비율 (기본 0.25 → 1/4 크기)
    """
    # 출력 폴더가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)

    # 지원할 이미지 확장자
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(exts):
            continue

        src_path = os.path.join(input_dir, fname)
        dst_path = os.path.join(output_dir, fname)

        try:
            with Image.open(src_path) as img:
                # 새 크기 계산
                new_w = int(img.width * scale)
                new_h = int(img.height * scale)

                # 리사이즈 (bicubic)
                img_resized = img.resize((new_w, new_h), resample=Image.BICUBIC)

                # 저장
                img_resized.save(dst_path)
                print(f"Saved: {dst_path} ({new_w}×{new_h})")
        except Exception as e:
            print(f"Error processing {src_path}: {e}")

if __name__ == "__main__":
    # 예시 사용법
    input_folder  = "../data/DIV2K_valid_HR"
    output_folder = "../data/no_use/DIV2K_valid_HR"
    downsample_images(input_folder, output_folder, scale=0.25)
