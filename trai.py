import cv2
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# chúng ta import các hàm cần thiết liên quan đến bài nhận diện khuôn mặt của mẫu code
# import các thư viện cần thiết tránh sai sót trong cách nhận diện khuôn mặt bị lỗi
# các hàm thư viện python có nhiều trên các trang web có thể tham khảo.


# Đường dẫn đến các thư mục chứa hình ảnh có khuôn mặt
positive_path = 'C:\\Users\\ADMIN\\PycharmProjects\\pythonProject1\\faces'


# Chức năng đọc và chuẩn bị dữ liệu
def read_images(path):
    images = []
    labels = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename))
        if img is not None:  # Kiểm tra xem hình ảnh đã được tải thành công chưa
            img = cv2.resize(img, (100, 100))  # Thay đổi kích thước hình ảnh thành kích thước mong muốn
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển đổi sang hình ảnh RGB
            img = img.astype('float32') / 255.0  # Chuẩn hóa giá trị pixel thành [0, 1]
            print("Shape of loaded image:", img.shape)  # In hình dạng của hình ảnh đã tải
            images.append(img)
            if "positive" in filename:
                labels.append(1)  # Nhãn 1 cho hình ảnh có khuôn mặt
            else:
                labels.append(0)  # Nhãn 0 cho hình ảnh không có khuôn mặt
        else:
            print(f"Error loading image: {os.path.join(path, filename)}")
    return np.array(images), np.array(labels)


# Đọc hình ảnh từ thư mục positive
X_positive, y_positive = read_images(positive_path)

# Khởi tạo mảng trống cho các ví dụ phủ định
X_negative = np.array([])
y_negative = np.array([])

# Kết hợp dữ liệu từ cả hai bộ dữ liệu
if X_positive.size > 0:  # Kiểm tra xem có tồn tại các ví dụ tích cực không
    X_train = X_positive
    y_train = y_positive
if X_negative.size > 0:  # Kiểm tra xem có tồn tại các ví dụ phủ định không
    X_train = np.concatenate((X_train, X_negative), axis=0)
    y_train = np.concatenate((y_train, y_negative), axis=0)


# Xây dựng mô hình CNN
# Các mô hình để xây dựng cnn để tạo ra các tích chập cnn trong hình ảnh cho các xác nhận của hình ảnh từ khuôn mặt
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Đào tạo mô hình
# Đâò tạo mô hình x và y để training laị cho khuôn mặt cần đươc xác minh danh tính tên tuổi con người
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Lưu mô hình vào một tệp
model.save('C:\\Users\\ADMIN\\PycharmProjects\\pythonProject1\\model\\face_detection_model.h5')
#file face_detection_model.h5
