import os

path = r"C:\Users\PAVAN H\Downloads\SIPaKMeD"

image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

print("=" * 70)
print("SIPaKMeD DATASET SUMMARY")
print("=" * 70)
print("Path:", path)

if not os.path.exists(path):
    print("\n❌ Dataset not found")
    exit()

for root, dirs, files in os.walk(path):

    images = [
        f for f in files
        if f.lower().endswith(image_extensions)
    ]

    if images:
        relative = os.path.relpath(root, path)

        print(f"\nFolder: {relative}")
        print(f"Images: {len(images)}")

print("\n" + "=" * 70)
print("SUMMARY COMPLETED")
print("=" * 70)