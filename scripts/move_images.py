import os
import shutil

SOURCE_DIR = "line_images"
DEST_DIR = "images"
TEXT_FILES = [
    "test_typewritten_best.txt",
    "train_typewritten_best.txt",
    "val_typewritten_best.txt",
]

os.makedirs(DEST_DIR, exist_ok=True)

filenames = set()
for txt in TEXT_FILES:
    with open(txt) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                print(parts[0])
                filenames.add(parts[0])

moved = 0
missing = 0
not_found = []
for name in filenames:
    src = os.path.join(SOURCE_DIR, name)
    dst = os.path.join(DEST_DIR, name)
    if os.path.exists(src):
        if os.path.exists(dst):
            os.remove(dst)
        shutil.move(src, dst)
        moved += 1
    else:
        not_found.append(name)
        missing += 1

if not_found:
    with open("notfound.txt", "w") as f:
        f.write("\n".join(sorted(not_found)) + "\n")

print(f"Moved: {moved}, Missing: {missing}")
