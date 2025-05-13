import os
import shutil
import random

# Paths
original_dataset_dir = r'C:\Users\gowri\PycharmProjects\Hematology\dataset'
base_dir = r'C:\Users\gowri\PycharmProjects\Hematology\split_data'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Create target directories
for split_path in [train_dir, test_dir]:
    os.makedirs(split_path, exist_ok=True)

# Accept all image types (case-insensitive)
image_extensions = ('.png', '.jpg', '.jpeg', '.jfif', '.tiff', '.bmp', '.webp', '.gif')

split_ratio = 0.8

for class_name in os.listdir(original_dataset_dir):
    class_path = os.path.join(original_dataset_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    print(f"\n📂 Processing class: {class_name}")

    contents = os.listdir(class_path)
    print(f"   📁 Found {len(contents)} items inside '{class_name}'")

    # Match ANY image file (case-insensitive)
    images = [img for img in contents if img.lower().endswith(image_extensions)]

    if not images:
        print(f"⚠ No valid image files found in class '{class_name}'. Skipping.")
        continue

    random.shuffle(images)
    split_index = int(len(images) * split_ratio)
    train_images = images[:split_index]
    test_images = images[split_index:]

    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    for img in train_images:
        shutil.copy2(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))

    for img in test_images:
        shutil.copy2(os.path.join(class_path, img), os.path.join(test_dir, class_name, img))

    print(f"   ✅ {len(train_images)} to train, {len(test_images)} to test")

print("\n🎉 Dataset split complete! Check your folders.")