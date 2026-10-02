import os
from PIL import Image

DATASETS = {
    "Mendeley LBC": r"C:\Users\PAVAN H\Downloads\Mendeley LBC Cervical Cancer",
    "Cytolog Cervical Cancer": r"C:\Users\PAVAN H\Downloads\Cytolog Cervical Cancer",
    "SIPaKMeD": r"C:\Users\PAVAN H\Downloads\SIPaKMeD",
    "CervicalCancer": r"C:\Users\PAVAN H\Downloads\CervicalCancer",
    "Custom Cervical Cancer Cytology Image": r"C:\Users\PAVAN H\Downloads\Custom Cervical Cancer Cytology Image",
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def inspect_dataset(name, path):
    print("\n" + "=" * 70)
    print(f"DATASET: {name}")
    print(f"PATH: {path}")
    print("=" * 70)

    if not os.path.exists(path):
        print("❌ Dataset folder not found!")
        return

    total_images = 0
    corrupted = 0
    extensions = {}
    dimensions = {}

    # Find folders containing images
    class_counts = {}

    for root, dirs, files in os.walk(path):
        image_files = [
            f for f in files
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ]

        if image_files:
            relative_path = os.path.relpath(root, path)

            # Treat the folder containing the images as the class folder
            class_name = relative_path.split(os.sep)[0]

            class_counts[class_name] = (
                class_counts.get(class_name, 0) + len(image_files)
            )

            for filename in image_files:
                file_path = os.path.join(root, filename)

                ext = os.path.splitext(filename)[1].lower()
                extensions[ext] = extensions.get(ext, 0) + 1

                total_images += 1

                try:
                    with Image.open(file_path) as img:
                        img.verify()

                    with Image.open(file_path) as img:
                        size = img.size
                        dimensions[size] = dimensions.get(size, 0) + 1

                except Exception:
                    corrupted += 1

    print(f"\nTotal images: {total_images}")
    print(f"Corrupted images: {corrupted}")

    print("\nClass / Folder Distribution:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name}: {count}")

    print("\nImage Formats:")
    for ext, count in sorted(extensions.items()):
        print(f"  {ext}: {count}")

    print("\nTop Image Dimensions:")
    for size, count in sorted(
        dimensions.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]:
        print(f"  {size}: {count}")


for name, path in DATASETS.items():
    inspect_dataset(name, path)

print("\n" + "=" * 70)
print("DATASET INSPECTION COMPLETED")
print("=" * 70)