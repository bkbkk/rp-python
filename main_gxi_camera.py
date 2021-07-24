import gxipy as gx
from PIL import Image
import cv2
from multiprocessing import Process, Manager
import datetime
import time
import onnxruntime as ort
from predictor import UltraPredictor
from detector import LightDetector
import numpy as np



    
class Producer(Process):
    def __init__(self, m, exposure=5000, width=960,
                 width_offset=160, height=1024, height_offset=0):
        super(Producer, self).__init__()
        self.m = m
        self.index = 0
        self.exposure=exposure
        self.width=width
        self.width_offset=width_offset
        self.height=height
        self.height_offset=height_offset
        self.m['shape'] = (width, height)
        
    def run(self):

        # create a device manager
        device_manager = gx.DeviceManager()
        dev_num, dev_info_list = device_manager.update_device_list()
        if dev_num == 0:
            print("Number of enumerated devices is 0")
            return
        # open the first device by exclusive mode
        cam ,handle= device_manager.open_device_by_index(1, 4)
        #设置缓冲区大小
        cam_datastream=gx.DataStream(handle)
        cam_datastream.set_acquisition_buffer_number(buf_num=1)
        # 设置参数
        # exit when the camera is a mono camera
        if cam.PixelColorFilter.is_implemented() is False:
            print("This sample does not support mono camera.")
            cam.close_device()
            return
        # set continuous acquisition
        if cam.AcquisitionMode.is_implemented() is True:
            print("the camera does support continue GxAcquisitionMode")
            cam.AcquisitionMode.set(gx.GxAcquisitionModeEntry.CONTINUOUS)
        else:
            print("the camera does not support continue GxAcquisitionMode")
        # set exposure
        cam.ExposureTime.set(self.exposure)
        # set gain
        cam.GainSelector.set(0)  # all:0
        cam.Gain.set(16)
        # set whitebalance
        cam.BalanceRatioSelector.set(0)  # red:0, green:1, blue:2
        cam.BalanceRatio.set(2)
        cam.BalanceRatioSelector.set(1) 
        cam.BalanceRatio.set(2)
        cam.BalanceRatioSelector.set(2)  
        cam.BalanceRatio.set(2)
        # set width
        cam.Width.set(self.width)
        cam.Height.set(self.height)
        # offset Y
        cam.OffsetY.set(self.height_offset)
        cam.OffsetX.set(self.width_offset)
        # set trigger mode and trigger source
        cam.TriggerMode.set(gx.GxSwitchEntry.OFF)
        cam.TriggerSource.set(gx.GxTriggerSourceEntry.SOFTWARE)
        
        
        print("相机的属性 ", 
              "\n width:        ", cam.Width.get(), 
              "\n Height:       ", cam.Height.get(),
              "\n exposure:     ", cam.ExposureTime.get(),
              "\n gain:         ", cam.Gain.get(),
              "\n TriggerMode:  ", cam.TriggerMode.get())
        cam.BalanceRatioSelector.set(0)
        print (" white balance red:    ", cam.BalanceRatio.get())
        cam.BalanceRatioSelector.set(1)
        print (" white balance green:  ", cam.BalanceRatio.get())
        cam.BalanceRatioSelector.set(2)
        print (" white balance blue:   ", cam.BalanceRatio.get())
        
        data_stream=cam.data_stream[0]
        # start data acquisition
        cam.stream_on()

        while True:
            if self.m['p'] >= self.m['c']:
                data_stream.flush_queue()
                self.index += 1
                raw_image = data_stream.get_image()
                ret=True
#                 print("Frame ID: %d   Height: %d   Width: %d"
#                       % (raw_image.get_frame_id(), 
#                          raw_image.get_height(), raw_image.get_width()))
                rgb_image = raw_image.convert("RGB")
                if rgb_image is None:
                    print('Failed to convert RawImage to RGBImage')
                    ret=False
                numpy_image = rgb_image.get_numpy_array()
                if numpy_image is None:
                    print('Failed to get numpy array from RGBImage')
                    ret=False
                img=Image.fromarray(numpy_image, 'RGB')
                frame=np.array(np.uint8(img))

                self.m['ret']   = ret
                self.m['frame'] = frame
                self.m['index'] = self.index
                self.m['p'] += 1

        # stop data acquisition
        cam.stream_off()
        # close device
        cam.close_device()




class Consumer(Process):
    def __init__(self, m, out_path,
                 label_path='./voc-model-labels.txt', 
                 onnx_path='./armor_slim_42_sim.onnx'):
        super(Consumer, self).__init__()
        self.m = m
        self.out_path = out_path
        self.predictor = None
        self.Ldetector = None
        self.Adetector = None
        
        self.class_names = [name.strip() for name in open(label_path).readlines()]
        self.ort_session = ort.InferenceSession(onnx_path)
        self.threshold   = 0.7
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
    def run(self):
        while self.m['shape'] is None:
            pass
        shape = self.m['shape']
        self.out       = cv2.VideoWriter(self.out_path, self.fourcc, 60, shape)
        self.Ldetector = LightDetector()
        self.predictor = UltraPredictor(self.ort_session, 
                                        self.threshold, self.class_names)
        while True:
            if self.m['c'] < self.m['p']:                
                ret   = self.m['ret']
                frame = self.m['frame']
                index = self.m['index']
                self.m['c'] += 1
                
                t1 = datetime.datetime.now()
                self.out.write(frame)
#                 self.predictor.predict(frame)
#                 self.Ldetector.set_img(frame)
#                 is_find = self.Ldetector.get_center()
        
#                 if is_find:
#                     cv2.drawContours(frame, [self.Ldetector.box], 0, (0,0,255), 2)
                    
#                 print ('frame index: ', index)
                cv2.imshow('frame', frame)
                if (cv2.waitKey(1) == 113): # q : quit
                    break
#                 print ('cost time: ', datetime.datetime.now()-t1, ' s')
        self.out.release()






if __name__ == '__main__':
    
    label_path = "./voc-model-labels.txt"
    onnx_path = "./armor_slim_42_sim.onnx"
    out_path = './save_video/' + \
    str(datetime.datetime.now()).replace(' ', '_').split('.')[0] + '.avi'
    
    m = Manager().dict()
    m['p'] = 0
    m['c'] = 0
    m['shape'] = None
    p1 = Producer(m)
    c1 = Consumer(m, out_path, label_path, onnx_path)
    
    p1.start()
    c1.start()
    p1.join()
    c1.join()
