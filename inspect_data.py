import cv2
import os
import numpy as np

def inspect_image(path, name):
    print(f"--- Inspecting {name} ---")
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    img = cv2.imread(path)
    if img is None:
        print("Could not read image.")
        return

    print(f"Shape: {img.shape}")
    print(f"Dtype: {img.dtype}")
    
    # Check if grayscale/binary or color
    is_grayscale = np.allclose(img[:,:,0], img[:,:,1]) and np.allclose(img[:,:,1], img[:,:,2])
    print(f"Is Grayscale/BW look: {is_grayscale}")

    # Check unique values to see if it's strictly binary
    unique_vals = np.unique(img)
    print(f"Unique pixel values count: {len(unique_vals)}")
    if len(unique_vals) < 10:
        print(f"Unique values: {unique_vals}")
    
    # Check background (Top-left corner 5x5)
    top_left = img[0:5, 0:5]
    avg_bg = np.mean(top_left)
    print(f"Top-Left 5x5 Mean: {avg_bg:.2f} (0=Black, 255=White)")
    
    if avg_bg < 50:
        print("-> Likely BLACK background")
    elif avg_bg > 200:
        print("-> Likely WHITE background")
    else:
        print("-> Mixed/Complex background")

# 1. Check processed Kaggle data
kaggle_dir = './data_kaggle'
# Find first available image
found = False
for root, dirs, files in os.walk(kaggle_dir):
    for file in files:
        if file.endswith('.jpg'):
            inspect_image(os.path.join(root, file), "Kaggle Processed Image")
            found = True
            break
    if found: break

# 2. Check original data
original_dir = './data'
# Check a 'fist' image
inspect_image(os.path.join(original_dir, 'fist_001.jpg'), "Original 'fist' Image")
# Check an 'L' image
inspect_image(os.path.join(original_dir, 'L_001.jpg'), "Original 'L' Image")
