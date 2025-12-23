import os
from PIL import Image

# 원본 이미지 폴더 경로
input_folder = '../data/BSD100'
# 결과 이미지 저장 경로
output_folder = 'results/test/pieonly/bicubic'

# 저장 폴더가 없으면 생성
os.makedirs(output_folder, exist_ok=True)

# 처리할 확장자 리스트
valid_exts = ['.png', '.jpg', '.jpeg', '.bmp']

for filename in os.listdir(input_folder):
    if os.path.splitext(filename)[1].lower() in valid_exts:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # 이미지 열기
        img = Image.open(input_path).convert("RGB")
        w, h = img.size

        # 1/4로 다운샘플링
        img_lowres = img.resize((w // 4, h // 4), resample=Image.BICUBIC)

        # 4배 업샘플링 (원래 크기)
        img_upsampled = img_lowres.resize((w, h), resample=Image.BICUBIC)

        # 저장
        img_upsampled.save(output_path)

print("완료!")
