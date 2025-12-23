import os

def rename_lr_to_hr(folder_path: str):
    """
    folder_path 폴더 내 모든 파일 이름에서 'LR'을 'HR'로 변경합니다.
    """
    for fname in os.listdir(folder_path):
        # 파일명이 LR을 포함하는 경우에만 처리
        if "LR" in fname:
            src = os.path.join(folder_path, fname)
            new_fname = fname.replace("LR", "HR")
            dst = os.path.join(folder_path, new_fname)
            try:
                os.rename(src, dst)
                print(f"Renamed: {fname} → {new_fname}")
            except Exception as e:
                print(f"Error renaming {fname}: {e}")

if __name__ == "__main__":
    image_folder = "results/test/pieonly/MSAESRGAN/BSD100"
    rename_lr_to_hr(image_folder)
