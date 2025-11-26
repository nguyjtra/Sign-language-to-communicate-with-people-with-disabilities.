import pandas as pd
import numpy as np
import cv2
import os
import shutil

# CẤU HÌNH
CSV_PATH = 'archive/sign_mnist_train.csv' 
OUTPUT_DIR = './data_kaggle' 
IMG_SIZE = 224 

def get_label_char(label_index):
    # Mapping số sang chữ cái
    if label_index >= 9: 
        label_index += 1
    return chr(label_index + 65)

def process_kaggle_data():
    print(f"--- Đang bắt đầu xử lý file {CSV_PATH} (V3 - Fix Lỗi Nền) ---")
    
    if not os.path.exists(CSV_PATH):
        print(f"LỖI: Không tìm thấy file tại '{CSV_PATH}'.")
        return

    # Xóa thư mục cũ để làm lại
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    print("Đang đọc dữ liệu...")
    df = pd.read_csv(CSV_PATH)
    
    labels = df['label'].values
    pixels = df.drop('label', axis=1).values

    print(f"-> Tìm thấy {len(labels)} ảnh. Bắt đầu xử lý...")

    count = 0
    for i, (label, row) in enumerate(zip(labels, pixels)):
        # 1. Reshape & Resize
        img = row.reshape(28, 28).astype('uint8')
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # 2. Threshold Otsu
        _, img_binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # === LOGIC V3: CHỈ KIỂM TRA 5 DÒNG TRÊN CÙNG ===
        # Lấy 5 dòng pixel đầu tiên (vùng trên đầu ảnh)
        top_region = img_binary[0:5, :]
        
        # Tính giá trị trung bình của vùng này
        # Nếu > 127 tức là đa số là màu Trắng (255) -> Nền Trắng -> Cần Đảo
        if np.mean(top_region) > 127:
            img_binary = cv2.bitwise_not(img_binary)
        # ===============================================

        # 3. Lưu ảnh
        char_label = get_label_char(label)
        folder_path = os.path.join(OUTPUT_DIR, char_label)
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        filename = f"{char_label}_kaggle_{i}.jpg"
        cv2.imwrite(os.path.join(folder_path, filename), img_binary)
        
        count += 1
        if count % 2000 == 0:
            print(f"Đã xử lý {count} ảnh...")

    print(f"--- HOÀN TẤT! Đã lưu {count} ảnh chuẩn Nền Đen - Tay Trắng ---")

if __name__ == "__main__":
    process_kaggle_data()