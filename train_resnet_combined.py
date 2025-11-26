import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# --- CẤU HÌNH ĐƯỜNG DẪN ---
KAGGLE_DIR = './data_kaggle'  # Thư mục chứa folder A, B, C...
ORIGINAL_DIR = './data'       # Thư mục chứa ảnh lẻ fist_..., L_...
MODEL_SAVE_PATH = 'models/resnet_model.h5'
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20

# 1. Định nghĩa Mapping cho dữ liệu cũ (Dựa trên file train cũ của bạn)
# 'fist' -> 'E' (chữ E trong thủ ngữ giống nắm đấm)
legacy_mapping = {
    'L_': 'L',
    'fi': 'E',
    'ok': 'F',
    'pe': 'V',
    'pa': 'B'
}

def load_combined_data():
    X = []
    y_labels = [] # Lưu nhãn dạng chữ (A, B, C...) trước
    
    print("--- BẮT ĐẦU LOAD DỮ LIỆU ---")

    # --- PHẦN 1: Load từ KAGGLE (Cấu trúc Folder) ---
    print(f"1. Đang đọc dữ liệu Kaggle từ {KAGGLE_DIR}...")
    kaggle_classes = [d for d in os.listdir(KAGGLE_DIR) if os.path.isdir(os.path.join(KAGGLE_DIR, d))]
    
    for category in kaggle_classes:
        path = os.path.join(KAGGLE_DIR, category)
        # Chỉ lấy các folder tên là chữ cái (A-Z)
        if len(category) == 1: 
            for img_name in os.listdir(path):
                try:
                    img_path = os.path.join(path, img_name)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # ResNet cần RGB
                        X.append(img)
                        y_labels.append(category) # Lưu nhãn là chữ cái (VD: 'A')
                except Exception:
                    pass

    # --- PHẦN 2: Load từ DATA GỐC (Cấu trúc Filename) ---
    print(f"2. Đang đọc dữ liệu gốc từ {ORIGINAL_DIR}...")
    # Kiểm tra file trong folder data
    for file in os.listdir(ORIGINAL_DIR):
        if file.endswith('.jpg') or file.endswith('.png'):
            # Xác định class dựa trên 2 chữ cái đầu (giống code cũ của bạn)
            prefix = file[0:2]
            
            # Nếu prefix nằm trong danh sách cần map (VD: 'fi' -> 'E')
            if prefix in legacy_mapping:
                mapped_label = legacy_mapping[prefix] # Lấy nhãn chuẩn (VD: 'E')
                
                path = os.path.join(ORIGINAL_DIR, file)
                img = cv2.imread(path)
                if img is not None:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    X.append(img)
                    y_labels.append(mapped_label) # Lưu nhãn chuẩn đã map
                    print(f"   + Đã gộp file cũ: {file} --> Class {mapped_label}")

    # --- XỬ LÝ DỮ LIỆU ---
    X = np.array(X, dtype='float32')
    X = preprocess_input(X) # Chuẩn hóa kiểu ResNet

    # Chuyển nhãn từ chữ sang số (A->0, B->1...)
    # Lấy danh sách tất cả các class duy nhất có trong dữ liệu
    unique_classes = sorted(list(set(y_labels)))
    print(f"-> Tổng hợp các class tìm thấy: {unique_classes}")
    
    # Tạo map từ Chữ -> Số
    label_map = {cls: i for i, cls in enumerate(unique_classes)}
    
    # Chuyển đổi list nhãn thành số
    y_indices = [label_map[label] for label in y_labels]
    y = to_categorical(y_indices, num_classes=len(unique_classes))
    
    return X, y, len(unique_classes), unique_classes

# --- THỰC THI ---
# 1. Load và gộp dữ liệu
X, y, num_classes, classes_list = load_combined_data()
print(f"Tổng số ảnh training: {X.shape[0]}")
print(f"Số lượng class: {num_classes}")

# 2. Chia Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Build Model ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# 4. Compile & Train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

print("Bắt đầu train model đa nguồn...")
history = model.fit(X_train, y_train, 
                    batch_size=BATCH_SIZE, 
                    epochs=EPOCHS, 
                    validation_data=(X_test, y_test),
                    callbacks=[checkpoint, early_stop])

# Lưu lại danh sách class để dùng cho file detection.py sau này
import pickle
with open("models/classes.pkl", "wb") as f:
    pickle.dump(classes_list, f)


# Vẽ biểu đồ
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.show()