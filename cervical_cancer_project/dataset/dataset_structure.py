import os

DATASET_PATHS = {
    "Mendeley LBC": r"C:\Users\PAVAN H\Downloads\Mendeley LBC Cervical Cancer",
    "Cytolog Cervical Cancer": r"C:\Users\PAVAN H\Downloads\Cytolog Cervical Cancer",
    "SIPaKMeD": r"C:\Users\PAVAN H\Downloads\SIPaKMeD",
    
}

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
)


def inspect_structure(name, path):

    print("\n" + "=" * 80)
    print(f"DATASET: {name}")
    print(f"PATH: {path}")
    print("=" * 80)

    if not os.path.exists(path):
        print("❌ PATH NOT FOUND")
        return

    print("\nFolder Structure:\n")

    for root, dirs, files in os.walk(path):

        level = root.replace(path, "").count(os.sep)
        indent = "    " * level

        folder_name = os.path.basename(root)

        image_count = sum(
            1 for file in files
            if file.lower().endswith(IMAGE_EXTENSIONS)
        )

        if image_count > 0:
            print(
                f"{indent}📁 {folder_name} "
                f"→ {image_count} images"
            )
        else:
            print(f"{indent}📁 {folder_name}")

        for file in files:

            if file.lower().endswith(IMAGE_EXTENSIONS):

                print(
                    f"{indent}    🖼 {file}"
                )


for name, path in DATASET_PATHS.items():
    inspect_structure(name, path)

print("\n" + "=" * 80)
print("DATASET STRUCTURE INSPECTION COMPLETED")
print("=" * 80)