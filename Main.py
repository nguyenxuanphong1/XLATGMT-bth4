import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from PIL import Image

# Đường dẫn tới thư mục chứa các Patch
dataset_path = "D:/XLATHGIMAYTINH/bth4/bth4/input/YTE"


# Tạo danh sách chứa ảnh và nhãn
data = []
labels = []

# Đọc ảnh từ các thư mục
for patch_num in range(1, 5):
    patch_path = os.path.join(dataset_path, f'Patch{patch_num}')
    if os.path.exists(patch_path):
        img_count = 0  # Đếm số lượng ảnh
        for img_name in os.listdir(patch_path):
            img_path = os.path.join(patch_path, img_name)
            if img_name.endswith(('.png', '.jpg', '.jpeg')):  # Kiểm tra định dạng ảnh
                # Mở và xử lý ảnh
                img = Image.open(img_path).convert('RGB')
                img = img.resize((64, 64))  # Đưa kích thước ảnh về 64x64 (hoặc kích thước bạn muốn)
                data.append(np.array(img).flatten())  # Chuyển ảnh thành mảng 1 chiều
                labels.append(f'Patch{patch_num}')  # Gán nhãn tương ứng
                img_count += 1
        print(f"Đã đọc {img_count} ảnh từ {patch_path}.")  # In số lượng ảnh đã đọc
    else:
        print(f"Lỗi: Không tìm thấy thư mục '{patch_path}'.")

# Chuyển đổi dữ liệu thành mảng numpy
X = np.array(data)
y = np.array(labels)

# Kiểm tra số lượng mẫu trong X
if X.size == 0:
    print("Lỗi: Không có ảnh nào được đọc. Vui lòng kiểm tra các thư mục.")
else:
    # Kịch bản chia dữ liệu (tỷ lệ tập test)
    patches = {
        "Patch 1 (80-20)": 0.2,
        "Patch 2 (70-30)": 0.3,
        "Patch 3 (60-40)": 0.4,
        "Patch 4 (40-60)": 0.6
    }

    # Khởi tạo danh sách để lưu trữ độ chính xác
    svm_accuracies = []
    knn_accuracies = []

    # Chạy thử nghiệm cho từng kịch bản
    for patch_name, test_size in patches.items():
        # Chia dữ liệu theo tỷ lệ của từng Patch
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Huấn luyện và đánh giá SVM
        svm_model = SVC()
        svm_model.fit(X_train, y_train)
        svm_predictions = svm_model.predict(X_test)
        svm_accuracy = accuracy_score(y_test, svm_predictions)
        svm_accuracies.append(svm_accuracy)

        # Huấn luyện và đánh giá KNN
        knn_model = KNeighborsClassifier()
        knn_model.fit(X_train, y_train)
        knn_predictions = knn_model.predict(X_test)
        knn_accuracy = accuracy_score(y_test, knn_predictions)
        knn_accuracies.append(knn_accuracy)

        # In kết quả cho từng kịch bản
        print(f"{patch_name} - Độ chính xác của SVM: {svm_accuracy:.2f}, Độ chính xác của KNN: {knn_accuracy:.2f}")

    # Nhận xét
    print("\nNhận xét:")
    for i, patch_name in enumerate(patches.keys()):
        print(f"{patch_name}:")
        if svm_accuracies[i] > knn_accuracies[i]:
            print("  - Mô hình SVM cho kết quả tốt hơn so với KNN.")
        elif svm_accuracies[i] < knn_accuracies[i]:
            print("  - Mô hình KNN cho kết quả tốt hơn so với SVM.")
        else:
            print("  - Cả hai mô hình đều cho kết quả tương đương.")

    # So sánh tổng thể
    avg_svm_accuracy = np.mean(svm_accuracies)
    avg_knn_accuracy = np.mean(knn_accuracies)
    print(f"\nĐộ chính xác trung bình của SVM: {avg_svm_accuracy:.2f}")
    print(f"Độ chính xác trung bình của KNN: {avg_knn_accuracy:.2f}")

    if avg_svm_accuracy > avg_knn_accuracy:
        print("Mô hình SVM có độ chính xác trung bình tốt hơn.")
    elif avg_svm_accuracy < avg_knn_accuracy:
        print("Mô hình KNN có độ chính xác trung bình tốt hơn.")
    else:
        print("Cả hai mô hình có độ chính xác trung bình tương đương.")
