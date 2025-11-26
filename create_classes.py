import pickle
import os

# Danh sách class đầy đủ dựa trên dữ liệu Kaggle và logic mapping
# A-Z trừ J, Z (24 ký tự)
classes = sorted(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'])

# Lưu file
os.makedirs("models", exist_ok=True)
with open("models/classes.pkl", "wb") as f:
    pickle.dump(classes, f)

print(f"Đã tạo file models/classes.pkl với {len(classes)} class: {classes}")
