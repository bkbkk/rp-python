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
        self.cap.set(cv2.CAP_PROP_EXPOSURE, 20)
        pos = self.cap.get(cv2.CAP_PROP_EXPOSURE)
        print (pos)
        
    def run(self):
        while True:
            if self.m['p'] >= self.m['c']:
                self.index += 1
                ret, frame = cap.read()
                
                self.m['ret']   = ret
                self.m['frame'] = frame
                self.m['index'] = self.index
#                 time.sleep(0.1)
                self.m['p'] += 1
#                 print ('send an data, index: ', self.index)


class Consumer(Process):
    def __init__(self, m, predictor, detector, out):
        super(Consumer, self).__init__()
        self.m = m
        self.predictor = predictor
        self.detector = detector
        self.out = out
        
    def run(self):
        while True:
            if self.m['c'] < self.m['p']:
                iscontinue = 0
                
                ret   = self.m['ret']
                frame = self.m['frame']
                index = self.m['index']
                self.m['c'] += 1
                
                t1 = datetime.datetime.now()
                self.out.write(frame)
#                 self.predictor.predict(frame)
#                 self.detector.set_img(frame)
#                 is_find = self.detector.get_center()
        
#                 if is_find:
#                     cv2.drawContours(frame, [self.detector.box], 0, (0,0,255), 2)
                    
                print ('frame index: ', index)
                cv2.imshow('frame', frame)
                cv2.waitKey(1)
                
#                 print ('cost time: ', datetime.datetime.now()-t1, ' s')
#                 print ('receive an data, index: ', index)






if __name__ == '__main__':
    cap = cv2.VideoCapture('/dev/video1')
    print ('cap open status: ', cap.isOpened())
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_EXPOSURE, 5)
    pos = cap.get(cv2.CAP_PROP_EXPOSURE)
    print ('pos: ', pos)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    buffer_size = cap.get(cv2.CAP_PROP_BUFFERSIZE)
    print ('buffer_size: ', buffer_size)
    rate = cap.get(cv2.CAP_PROP_FPS)
    print ('rate: ', rate)
    
    shape = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print ('cap frame shape: ', shape)
    save_video_name = './save_video/' + str(datetime.datetime.now()).replace(' ', '_').split('.')[0] + '.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(save_video_name, fourcc, 60, shape)
    
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
    c1 = Consumer(m, predictor, detector, out)
    
    p1.start()
    c1.start()
    p1.join()
    c1.join()
