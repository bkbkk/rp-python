import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from multiprocessing import Process, Manager
import datetime
import time
import onnxruntime as ort
from predictor import UltraPredictor
from detector import LightDetector
import numpy as np


class Producer(Process):
    def __init__(self, m, cap):
        super(Producer, self).__init__()
        self.m = m
        self.index = 0
        self.cap = cap
        self.start_frame = 850
        self.total_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         self.cap.set(cv2.CAP_PROP_EXPOSURE, 60)
        
    def run(self):
        while True:
            if (self.m['p'] >= self.m['c']) and (self.m['p'] < self.m['c']+self.start_frame):
                self.index += 1
                ret, frame = cap.read()
                
                self.m['ret']   = ret
                self.m['frame'] = frame
                self.m['index'] = self.index
#                 time.sleep(0.1)
                self.m['p'] += 1
#                 print ('send an data, index: ', self.index)


class Consumer(Process):
    def __init__(self, m, predictor, detector):
        super(Consumer, self).__init__()
        self.m = m
        self.predictor = predictor
        self.detector = detector
        
    def run(self):
        while True:
            if self.m['c'] < self.m['p']:
                iscontinue = 0
                
                ret   = self.m['ret']
                frame = self.m['frame']
                index = self.m['index']
                self.m['c'] += 1
                
                t1 = datetime.datetime.now()
#                 self.predictor.predict(frame)
                self.detector.set_img(frame)
                is_find = self.detector.get_center()
                if is_find:
                    cv2.drawContours(frame, [self.detector.box], 0, (0,0,255), 2)
                    x, y = self.detector.center[0], self.detector.center[1]
                    print ('x: ', x, ' y: ', y)
                    print (type(x))
                
                print ('frame index: ', index)
                cv2.imshow('frame', frame)
                key = cv2.waitKey(0)
                if key == 113:  # q : quit
                    break
                    
        cv2.destroyAllWindows()
        self.cap.release()


#                 print ('cost time: ', datetime.datetime.now()-t1, ' s')
                
#                 print ('receive an data, index: ', index)



if __name__ == '__main__':
    cap = cv2.VideoCapture('./drone/greenlight.avi')
#     cap.set(cv2.CAP_PROP_POS_FRAMES, 840)
    label_path = "./voc-model-labels.txt"
    onnx_path = "./armor_slim_42_sim.onnx"
    class_names = [name.strip() for name in open(label_path).readlines()]
    ort_session = ort.InferenceSession(onnx_path)
    threshold = 0.7
    predictor = UltraPredictor(ort_session, threshold, class_names)
    detector = LightDetector()
    
    m = Manager().dict()
    m['p'] = 0
    m['c'] = 0
    p1 = Producer(m, cap)
    c1 = Consumer(m, predictor, detector)

    p1.start()
    c1.start()
    p1.join()
    c1.join()