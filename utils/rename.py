import os

def rename_lr_to_hr(folder_path: str):
    """
    Change 'LR' to 'HR' in all filenames within folder_path folder.
    """
    for fname in os.listdir(folder_path):
        # Process only if filename contains LR
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
