import cv2
import numpy as np
import pickle
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# --- CẤU HÌNH ---
MODEL_PATH = 'models/resnet_model.h5'
CLASS_PATH = 'models/classes.pkl'
THRESHOLD_SCORE = 0.85  # Độ tin cậy tối thiểu (85%)
STABILITY_FRAMES = 15   # Phải giữ yên tay trong 15 khung hình (khoảng 0.5s) mới nhận chữ
BOX_SIZE = 224          # Kích thước vùng nhận diện

# --- LOAD MODEL & CLASS ---
print("Đang load model & danh sách class...")
model = load_model(MODEL_PATH)

try:
    with open(CLASS_PATH, 'rb') as f:
        classes = pickle.load(f)
    print(f"Đã load {len(classes)} class: {classes}")
except:
    print("LỖI: Không tìm thấy file classes.pkl. Hãy chắc chắn bạn đã chạy file train_resnet_combined.py")
    exit()

# Biến toàn cục xử lý logic gõ chữ
current_word = ""
last_pred_char = ""
stable_count = 0

# Biến xử lý camera & nền
background = None
is_bg_captured = False

def get_prediction(img_binary):
    # 1. Resize về 224x224
    img = cv2.resize(img_binary, (BOX_SIZE, BOX_SIZE))
    
    # 2. ResNet cần ảnh màu (3 kênh), ta chồng ảnh binary lên 3 lớp
    img_stack = np.stack((img,)*3, axis=-1)
    
    # 3. Chuẩn hóa (Preprocess)
    img_input = img_stack.astype('float32')
    img_input = preprocess_input(img_input) # Hàm chuẩn của Keras
    img_input = np.expand_dims(img_input, axis=0) # Thêm chiều batch: (1, 224, 224, 3)
    
    # 4. Dự đoán
    preds = model.predict(img_input, verbose=0)
    idx = np.argmax(preds)
    score = np.max(preds)
    
    return classes[idx], score

# --- CHƯƠNG TRÌNH CHÍNH ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("\n--- HƯỚNG DẪN SỬ DỤNG ---")
print("1. Đưa camera vào vị trí cố định.")
print("2. Bấm phím 'B' để chụp lại nền (QUAN TRỌNG: Không để tay trong khung hình lúc bấm B).")
print("3. Đưa tay vào ô vuông xanh để nhận diện.")
print("4. Giữ yên tay để nhập chữ.")
print("5. Bấm 'R' để reset chữ đã gõ. 'Q' để thoát.\n")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1) # Lật gương
    h, w, _ = frame.shape
    
    # Vẽ ô vuông vùng nhận diện (ROI)
    # Đặt lệch sang phải một chút để dễ nhìn
    roi_top, roi_right = 50, w - 50
    roi_bottom, roi_left = 300, w - 300
    cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), 2)
    
    # Cắt vùng ảnh ROI
    roi = frame[roi_top:roi_bottom, roi_left:roi_right]
    
    if is_bg_captured:
        # --- XỬ LÝ TÁCH NỀN (Background Subtraction) ---
        # 1. Làm mờ nhẹ để giảm nhiễu
        roi_blur = cv2.GaussianBlur(roi, (7, 7), 0)
        bg_blur = cv2.GaussianBlur(background, (7, 7), 0)
        
        # 2. Tính hiệu số giữa nền và ảnh hiện tại
        diff = cv2.absdiff(bg_blur, roi_blur)
        
        # 3. Chuyển sang ảnh xám
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # 4. Phân ngưỡng (Threshold) để tách tay ra khỏi nền
        # Những điểm khác biệt lớn (tay) sẽ thành màu Trắng, nền thành Đen
        _, thresh = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Hiển thị ảnh Binary để debug (Ảnh này phải là Tay Trắng - Nền Đen)
        cv2.imshow("Binary View (AI Input)", thresh)
        
        # --- NHẬN DIỆN ---
        # Chỉ nhận diện nếu vùng tay đủ lớn (tránh nhiễu)
        if np.count_nonzero(thresh) > 5000: 
            char, score = get_prediction(thresh)
            
            if score > THRESHOLD_SCORE:
                # Hiển thị kết quả tạm thời trên đầu ô vuông
                cv2.putText(frame, f"{char} ({score*100:.1f}%)", (roi_left, roi_top - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                # Logic gõ chữ (Debounce)
                if char == last_pred_char:
                    stable_count += 1
                else:
                    stable_count = 0
                    last_pred_char = char
                
                # Nếu giữ yên tay đủ lâu -> Chấp nhận ký tự
                if stable_count == STABILITY_FRAMES:
                    # Nếu là ký tự đặc biệt (ví dụ 'SPACE' hoặc 'POINT') thì xử lý riêng
                    if char == 'POINT' or char == 'SPACE':
                        current_word += " " # Thêm dấu cách
                    else:
                        current_word += char
                    
                    stable_count = 0 # Reset để chờ ký tự tiếp
            
    # --- GIAO DIỆN ---
    # Vẽ thanh gõ chữ màu đen
    cv2.rectangle(frame, (0, h-80), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, f"Text: {current_word}", (20, h-30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    # Hướng dẫn
    if not is_bg_captured:
        cv2.putText(frame, "Bam 'B' de chup Background (Bo tay ra khoi khung hinh)", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("Hand Sign App", frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('b'):
        # Chụp lại nền tại vùng ROI
        background = roi.copy()
        is_bg_captured = True
        print("Đã chụp background thành công!")
    elif key == ord('r'):
        current_word = "" # Xóa chữ làm lại

cap.release()
cv2.destroyAllWindows()