import cv2
import face_recognition
import numpy as np
from keras.models import load_model

# Tải mô hình phát hiện khuôn mặt được đào tạo sẵn
# Sau khi đào tạo mô hình khuôn mặt thì chúng ta khởi tạo khuôn mặt đã được đào tạo lên để đưa vào camera để được xác minh
model = load_model('C:\\Users\\ADMIN\\PycharmProjects\\pythonProject1\\model\\face_detection_model.h5')

# Tải bộ phân loại Haar Cascade để nhận diện khuôn mặt
# tải file haarcascade_frontalfac_default.xml để đọc được mã code
face_cascade = cv2.CascadeClassifier('C:\\Users\\ADMIN\\PycharmProjects\\pythonProject1\\haarcascades\\haarcascade_frontalface_default.xml')

# Tải hình ảnh của các cá nhân và tên tương ứng của họ
# Tải hình ảnh và đối chiếu hình ảnh của mình cho các tên tương ứng cần thiết đã code từ trước
luongpham_image = face_recognition.load_image_file('C:\\Users\\ADMIN\\PycharmProjects\\pythonProject1\\faces\\luongpham.png')
ronaldo_image = face_recognition.load_image_file('C:\\Users\\ADMIN\\PycharmProjects\\pythonProject1\\faces\\ronaldo.png')
empape_image = face_recognition.load_image_file('C:\\Users\\ADMIN\\PycharmProjects\\pythonProject1\\faces\\empape.png')
benzema_image = face_recognition.load_image_file('C:\\Users\\ADMIN\\PycharmProjects\\pythonProject1\\faces\\bezema.png')
bale_image = face_recognition.load_image_file('C:\\Users\\ADMIN\\PycharmProjects\\pythonProject1\\faces\\bale.png')
lucas_image = face_recognition.load_image_file('C:\\Users\\ADMIN\\PycharmProjects\\pythonProject1\\faces\\lucas.png')
modric_image = face_recognition.load_image_file('C:\\Users\\ADMIN\\PycharmProjects\\pythonProject1\\faces\\modric.png')
pepe_image = face_recognition.load_image_file('C:\\Users\\ADMIN\\PycharmProjects\\pythonProject1\\faces\\pepe.png')
romus_image = face_recognition.load_image_file('C:\\Users\\ADMIN\\PycharmProjects\\pythonProject1\\faces\\romus.png')

# Tính toán mã hóa khuôn mặt
# Đọc mã hóa khuôn mặt và đưa khuôn mặt sao cho đúng khung hình cần nhận diện khuôn mặt
luongpham_encoding = face_recognition.face_encodings(luongpham_image)[0]
ronaldo_encoding = face_recognition.face_encodings(ronaldo_image)[0]
empape_encoding = face_recognition.face_encodings(empape_image)[0]
benzema_encoding = face_recognition.face_encodings(benzema_image)[0]
bale_encoding = face_recognition.face_encodings(bale_image)[0]
lucas_encoding = face_recognition.face_encodings(lucas_image)[0]
modric_encoding = face_recognition.face_encodings(modric_image)[0]
pepe_encoding = face_recognition.face_encodings(pepe_image)[0]
romus_encoding = face_recognition.face_encodings(romus_image)[0]

# Danh sách các khuôn mặt và tên tương ứng
# Cần đưa ra các tên tương ứng cho từng hình ảnh và khuôn mặt chỉ có vậy mới đảm bảo được rằng khuôn mặt được xác mình là đúng
known_faces = [
    luongpham_encoding, ronaldo_encoding, empape_encoding, benzema_encoding,
    bale_encoding, lucas_encoding, modric_encoding, pepe_encoding, romus_encoding
]
# tên người trong khung ảnh
names = [
    'Pham Trong Luong', 'Ronaldo', 'Empape', 'Benzema',
    'Bale', 'Lucas', 'Modric', 'Pepe', 'Romus'
]

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Tìm các khuôn mặt trong khung hình
    # Tìm các khuôn mặt tương ứng và open nhận diện
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Duyệt qua từng khuôn mặt và nhận dạng
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # So sánh khuôn mặt hiện tại với danh sách các khuôn mặt đã biết
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"  # Mặc định là Unknown (sai)

        # Nếu có kết quả trùng khớp, hiển thị tên tương ứng
        if True in matches:
            first_match_index = matches.index(True)
            name = names[first_match_index]

        # Vẽ hộp xung quanh khuôn mặt và hiển thị tên
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Hiển thị khung hình với kết quả nhận dạng
    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng bộ nhớ và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()

#end và bài làm kết thúc
