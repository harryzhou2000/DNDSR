import os
import shutil


paths = []

dstPath = "../build/docs/html"

if not os.path.exists(dstPath):
    os.makedirs(dstPath)

for current_dir, subdirs, files in os.walk("."):
    # Skip subdirs since we're only interested in files.
    for filename in files:
        if filename.endswith((".pdf", ".png", ".jpg")):
            relative_path = os.path.join(current_dir, filename)
            absolute_path = os.path.abspath(relative_path)

            paths.append(relative_path)
            if not os.path.exists(os.path.join(dstPath, current_dir)):
                os.makedirs(os.path.join(dstPath, current_dir))
            shutil.copy2(relative_path, os.path.join(dstPath, relative_path))
            # print("from "+ relative_path)
            # print("to "+ os.path.join(dstPath, relative_path))

print(paths)
