import argparse
import datetime
import cv2
import imutils
from detector.yolo import YOLO

# Tham số đầu vào
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="Nguồn video mặc định 0 là sử dụng webcam", default='0', type=str)
args = vars(ap.parse_args())
detector_yolo = YOLO(root_dir='models', model_type='')
# Chọn nguồn video đầu vào
if args['video'] == '0':
    capture = cv2.VideoCapture(0)
else:
    capture = cv2.VideoCapture(args['video'])

labels = detector_yolo.labels

idx = 0
#  Xác định vùng quan sát trên ảnh
top_left, bottom_right = (330, 50), (450, 280)

out = None

#  Lặp lần lượt từng frame của video
while True:
    ret, frame = capture.read()
    if frame is None:
        break
    text = "An toan"
    frame = imutils.resize(frame, width=500)
    if out is None:
        frame_height, frame_width = frame.shape[:2]
        out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
    img, boxes, confidences, classids, idxs = detector_yolo.detector(frame)
    # Xác định vùng cần kiểm tra
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
    for i in idxs.flatten():
        if labels[classids[i]] == 'person':
            # Lấy tọa độ của hình chữ nhật bao quanh đối tượng
            (x, y, w, h) = boxes[i]
            # Xác định tâm của đối tượng
            center_x = x + w / 2
            center_y = y + h / 2
            # Kiểm tra đối tượng có nằm trong khu vực quan sát hay không
            logic = top_left[0] < center_x < bottom_right[0] and top_left[1] < center_y < bottom_right[1]
            if logic:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                text = "Co xam nhap"
                # Hiện cảnh báo lên hình
                cv2.putText(frame, "Tinh trang: {}".format(text), (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                text = "{}: {:4f}".format(labels[classids[i]], confidences[i])
                cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    out.write(frame)
        # show the frame and record if the user presses a key
    cv2.imshow("Camera an ninh", frame)
    # cv2.imshow("Thresh", thresh)
    cv2.imwrite('../images/{}_result.jpg'.format(idx), frame)
    # cv2.imwrite('../images/{}_delta.jpg'.format(idx), frameDelta)
    # cv2.imshow("Frame Delta", frameDelta)
    idx += 1
    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
cv2.destroyAllWindows()
