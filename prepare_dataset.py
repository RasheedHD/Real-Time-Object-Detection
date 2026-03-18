from pathlib import Path
import random
import shutil

TRAIN_RATIO = 0.8
OUTPUT_DIR = "dataset"
CLASS_NAMES = ["animal"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SEED = 42

def main():
    random.seed(SEED)

    project_root = Path(".")
    output = Path(OUTPUT_DIR)

    # Find all images recursively
    image_files = []
    for ext in IMAGE_EXTS:
        image_files.extend(project_root.rglob(f"*{ext}"))
        image_files.extend(project_root.rglob(f"*{ext.upper()}"))

    # Exclude already-generated dataset folders
    image_files = [
        p for p in image_files
        if OUTPUT_DIR not in p.parts and p.name.lower() != "best.pt"
    ]

    pairs = []
    for img_path in image_files:
        label_path = img_path.with_suffix(".txt")
        if label_path.exists():
            pairs.append((img_path, label_path))

    if not pairs:
        print("No image-label pairs found.")
        print("Check where your dataset files actually are.")
        return

    print(f"Found {len(pairs)} image-label pairs.")

    # Create folder structure
    for split in ["train", "val"]:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    random.shuffle(pairs)
    train_count = int(len(pairs) * TRAIN_RATIO)

    train_pairs = pairs[:train_count]
    val_pairs = pairs[train_count:]

    for split_name, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        for img_path, label_path in split_pairs:
            shutil.copy2(img_path, output / "images" / split_name / img_path.name)
            shutil.copy2(label_path, output / "labels" / split_name / label_path.name)

    yaml_content = f"""path: {OUTPUT_DIR}
train: images/train
val: images/val

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    Path("data.yaml").write_text(yaml_content, encoding="utf-8")

    print("Done.")
    print(f"Train: {len(train_pairs)}")
    print(f"Val: {len(val_pairs)}")
    print("Created dataset/ and data.yaml")

if __name__ == "__main__":
    main()