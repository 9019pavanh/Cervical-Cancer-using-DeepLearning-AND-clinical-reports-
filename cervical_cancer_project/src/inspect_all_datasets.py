from pathlib import Path
from PIL import Image
from collections import Counter


try:
    from dataset_config import DATASETS as CFG_DATASETS
    DATASETS = {name: cfg.active_path for name, cfg in CFG_DATASETS.items()}
except ImportError:
    try:
        from src.dataset_config import DATASETS as CFG_DATASETS
        DATASETS = {name: cfg.active_path for name, cfg in CFG_DATASETS.items()}
    except ImportError:
        DATASETS = {
            "Mendeley_LBC": Path(
                r"C:\Users\PAVAN H\Downloads\Mendeley LBC Cervical Cancer"
            ),
            "Cytolog": Path(
                r"C:\Users\PAVAN H\Downloads\Cytolog Cervical Cancer\Image"
            ),
            "SIPaKMeD": Path(
                r"C:\Users\PAVAN H\Downloads\SIPaKMeD"
            ),
            "CervicalCancer": Path(
                r"C:\Users\PAVAN H\Downloads\CervicalCancer"
            ),
            "Custom_Cervical_Cancer_Cytology": Path(
                r"C:\Users\PAVAN H\Downloads\Custom Cervical Cancer Cytology Image\Renamed_Custom Cytology Dataset"
            ),
        }


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def inspect_dataset(name, root):

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    print(f"Path: {root}")

    if not root.exists():
        print("ERROR: Dataset path does not exist.")
        return

    class_counts = Counter()
    dimensions = Counter()
    corrupted = []

    total_images = 0

    for file in root.rglob("*"):

        if not file.is_file():
            continue

        if file.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        total_images += 1

        # Check image
        try:

            with Image.open(file) as img:
                img.verify()

            with Image.open(file) as img:
                dimensions[img.size] += 1

        except Exception:

            corrupted.append(str(file))

        # Determine class name
        relative_path = file.relative_to(root)

        if name == "CervicalCancer" and len(relative_path.parts) >= 3:
            class_name = f"{relative_path.parts[0]} / {file.parent.name}"
        elif len(relative_path.parts) >= 2:
            class_name = relative_path.parts[0]
        else:
            class_name = file.parent.name if file.parent != root else "ROOT_LEVEL"

        class_counts[class_name] += 1

    print(f"\nTotal images: {total_images}")

    print(f"Corrupted images: {len(corrupted)}")

    print("\nClasses:")

    for class_name, count in sorted(class_counts.items()):

        print(f"  {class_name}: {count}")

    print(f"\nImage dimensions (Top 5 of {len(dimensions)} unique sizes):")

    for dimension, count in dimensions.most_common(5):

        print(f"  {dimension}: {count}")

    if corrupted:

        print("\nCorrupted files:")

        for file in corrupted[:20]:

            print(f"  {file}")


def main():

    print("\n" + "#" * 80)
    print("CERVICAL CANCER PROJECT - COMPLETE DATASET INSPECTION")
    print("#" * 80)

    for name, path in DATASETS.items():

        inspect_dataset(name, path)

    print("\n" + "#" * 80)
    print("INSPECTION COMPLETE")
    print("#" * 80)


if __name__ == "__main__":

    main()
    