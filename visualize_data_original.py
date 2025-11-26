import os
import matplotlib.pyplot as plt

# --- CẤU HÌNH ---
ORIGINAL_DIR = './data'

# Mapping cho dữ liệu cũ
legacy_mapping = {
    'L_': 'L',
    'fi': 'E',
    'ok': 'F',
    'pe': 'V',
    'pa': 'B'
}

def get_original_data_distribution():
    class_counts = {}

    print(f"Đang quét dữ liệu từ: {ORIGINAL_DIR}")
    if os.path.exists(ORIGINAL_DIR):
        for file in os.listdir(ORIGINAL_DIR):
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                # Xác định class dựa trên 2 chữ cái đầu
                prefix = file[0:2]
                
                if prefix in legacy_mapping:
                    label = legacy_mapping[prefix]
                    class_counts[label] = class_counts.get(label, 0) + 1
                else:
                    # Nếu file không khớp prefix nào trong map, có thể in ra để debug hoặc bỏ qua
                    pass
    else:
        print(f"Không tìm thấy thư mục {ORIGINAL_DIR}")

    return class_counts

def plot_distribution(class_counts):
    if not class_counts:
        print("Không tìm thấy ảnh nào khớp với mapping cũ!")
        return

    labels = sorted(class_counts.keys())
    counts = [class_counts[l] for l in labels]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, counts, color='lightcoral') # Màu khác để phân biệt
    
    plt.xlabel('Class (Hand Sign)')
    plt.ylabel('Number of Images')
    plt.title('Original Data Distribution (Old Dataset)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add count labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, int(yval), ha='center', va='bottom')

    plt.tight_layout()
    output_file = 'data_distribution_original.png'
    plt.savefig(output_file)
    print(f"Đã lưu biểu đồ vào: {output_file}")
    plt.show()

if __name__ == "__main__":
    counts = get_original_data_distribution()
    print(f"Kết quả đếm: {counts}")
    plot_distribution(counts)
