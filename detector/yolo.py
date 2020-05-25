import cv2 as cv
import numpy as np
from detector.yolo_utils import detector

class YOLO:

    def __init__(self, root_dir='../models', confidence=0.5, threshold=0.5, model_type=''):
        self.root_dir = root_dir
        self.model_type = model_type
        self.label_file = '{}/config/coco.names'.format(self.root_dir)
        self.weights_path = '{}/weights/yolov3{}.weights'.format(self.root_dir, self.model_type)
        self.config_path = '{}/config/yolov3{}.cfg'.format(self.root_dir, self.model_type)
        self.confidence = confidence
        self.threshold = threshold
        self.labels = open(self.label_file).read().strip().split('\n')
        self.net = cv.dnn.readNetFromDarknet(self.config_path, self.weights_path)
        # Get the output layer names of the model
        self.layer_names = self.net.getLayerNames()
        self.layer_names = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def detector(self, image, is_draw=False):
        height, width = image.shape[:2]
        colors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype='uint8')
        img, boxes, confidences, classids, idxs = detector(self.net, self.layer_names, height, width, image, colors,
                                                           self.labels, confidence=self.confidence,
                                                           threshold=self.threshold, is_draw=is_draw)
        return img, boxes, confidences, classids, idxs


if __name__ == '__main__':
    detector_yolo = YOLO()
    # Infer real-time on webcam
    vid = cv.VideoCapture('E:\\01. Project\Python\\aka-view\\video_test\TestVideo.mp4')
    while True:
        _, frame = vid.read()
        height, width = frame.shape[:2]
        img, boxes, confidences, classids, idxs = detector_yolo.detector(frame, is_draw=True)

        cv.imshow('webcam', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv.destroyAllWindows()
