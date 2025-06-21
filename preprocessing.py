import numpy as np
import cv2
import os
from image_processing import func

# Create output directories
if not os.path.exists("data2"):
    os.makedirs("data2")
if not os.path.exists("data2/train"):
    os.makedirs("data2/train")
if not os.path.exists("data2/test"):
    os.makedirs("data2/test")

path = "data/train"
path1 = "data2"
label = 0
var = 0
c1 = 0
c2 = 0

# Check if the train directory exists
if not os.path.exists(path):
    print(f"Directory '{path}' does not exist.")
else:
    print(f"Contents of '{path}': {os.listdir(path)}")

# Walk through the train directory
for (dirpath, dirnames, filenames) in os.walk(path):
    print(f"Current directory: {dirpath}")
    print(f"Subdirectories: {dirnames}")
    print(f"Files: {filenames}")

    for dirname in dirnames:
        print(f"Processing directory: {dirname}")
        for (direcpath, direcnames, files) in os.walk(os.path.join(path, dirname)):
            if not os.path.exists(os.path.join(path1, "train", dirname)):
                os.makedirs(os.path.join(path1, "train", dirname))
            if not os.path.exists(os.path.join(path1, "test", dirname)):
                os.makedirs(os.path.join(path1, "test", dirname))

            num = 100000000000000000  # Large number to save all to train
            i = 0

            for file in files:
                var += 1
                actual_path = os.path.join(path, dirname, file)
                actual_path1 = os.path.join(path1, "train", dirname, file)
                actual_path2 = os.path.join(path1, "test", dirname, file)

                img = cv2.imread(actual_path, 0)
                if img is None:
                    print(f"Failed to read image: {actual_path}")
                    continue

                # Uncomment the next line to use the actual processing function
                # bw_image = func(actual_path)
                bw_image = img  # Temporarily use the original image for testing

                if i < num:
                    c1 += 1
                    cv2.imwrite(actual_path1, bw_image)
                else:
                    c2 += 1
                    cv2.imwrite(actual_path2, bw_image)

                i += 1

        label += 1

print(f"Total images processed: {var}")
print(f"Images saved to train: {c1}")
print(f"Images saved to test: {c2}")
