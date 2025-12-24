import os
from PIL import Image

# Original image folder path
input_folder = '../data/BSD100'
# Result image save path
output_folder = 'results/test/pieonly/bicubic'

# Create save folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# List of extensions to process
valid_exts = ['.png', '.jpg', '.jpeg', '.bmp']

for filename in os.listdir(input_folder):
    if os.path.splitext(filename)[1].lower() in valid_exts:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Open image
        img = Image.open(input_path).convert("RGB")
        w, h = img.size

        # Downsample by 1/4
        img_lowres = img.resize((w // 4, h // 4), resample=Image.BICUBIC)

        # Upsample by 4x (original size)
        img_upsampled = img_lowres.resize((w, h), resample=Image.BICUBIC)

        # Save
        img_upsampled.save(output_path)

print("completed!")
