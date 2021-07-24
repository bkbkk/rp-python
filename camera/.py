import gxipy as gx
# import sys
# import time
from PIL import Image
import cv2
import numpy as np
# import threading

# import sys
import cv2
from multiprocessing import Process, Manager
import datetime
import time
import onnxruntime as ort
from predictor import UltraPredictor
from detector import LightDetector
import numpy as np

# def capture_callback_color(raw_image):
#     # print height, width, and frame ID of the acquisition image
#     print("Frame ID: %d   Height: %d   Width: %d"
#           % (raw_image.get_frame_id(), raw_image.get_height(), raw_image.get_width()))

#     # get RGB image from raw image
#     rgb_image = raw_image.convert("RGB")
#     if rgb_image is None:
#         print('Failed to convert RawImage to RGBImage')
#         return

#     # create numpy array with data from rgb image
#     numpy_image = rgb_image.get_numpy_array()
#     if numpy_image is None:
#         print('Failed to get numpy array from RGBImage')
#         return


#     # show acquired image
#     img = Image.fromarray(numpy_image, 'RGB')
#     cv2.imshow("aaa",np.array(np.uint8(img)))
#     cv2.waitKey(1)

    
class Producer(Process):
    def __init__(self, m, exposure=2000,width=1280,
                 width_offset=0,height=720,height_offset=304):
        super(Producer, self).__init__()
        self.m = m
        self.index = 0
        self.exposure=exposure
        self.width=width
        self.width_offset=width_offset
        self.height=height
        self.height_offset=height_offset
        
    def run(self):
        print("")
        print("-------------------------------------------------------------")
        print("Sample to show how to acquire color image continuously and show acquired image.")
        print("-------------------------------------------------------------")
        print("")
        print("Initializing......")
        print("")

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
        # cam.Gain.set(10.0)

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
              " width:       ", cam.Width.get(), 
              " Height:      ", cam.Height.get(),
              " exposure:    ", cam.ExposureTime.get(), 
              " TriggerMode: ", cam.TriggerMode.get())
        # # get param of improving image quality
        # if cam.GammaParam.is_readable():
        #     gamma_value = cam.GammaParam.get()
        #     gamma_lut = gx.Utility.get_gamma_lut(gamma_value)
        # else:
        #     gamma_lut = None
        # if cam.ContrastParam.is_readable():
        #     contrast_value = cam.ContrastParam.get()
        #     contrast_lut = gx.Utility.get_contrast_lut(contrast_value)
        # else:
        #     contrast_lut = None
        # if cam.ColorCorrectionParam.is_readable():
        #     color_correction_param = cam.ColorCorrectionParam.get()
        # else:
        #     color_correction_param = 0
        
        data_stream=cam.data_stream[0]
#         if cam.PixelColorFilter.is_implemented() is True:
#             data_stream.register_capture_callback(capture_callback_color)
#         else:
#             print("getting color picture failed ")

        # start data acquisition
        cam.stream_on()

        while True:
            while (self.m['p'] - self.m['c'] > 1):
                data_stream.flush_queue()
                cv2.waitKey(2)
            self.index += 1
            raw_image = data_stream.get_image()
            ret=True
                
#             raw_image=data_stream.get_image()
            print("Frame ID: %d   Height: %d   Width: %d"
                  % (raw_image.get_frame_id(), 
                     raw_image.get_height(), raw_image.get_width()))

            # get RGB image from raw image
            rgb_image = raw_image.convert("RGB")
            if rgb_image is None:
                print('Failed to convert RawImage to RGBImage')
                ret=False

            # create numpy array with data from rgb image
            numpy_image = rgb_image.get_numpy_array()
            if numpy_image is None:
                print('Failed to get numpy array from RGBImage')
                ret=False
                
            img=Image.fromarray(numpy_image, 'RGB')
            frame=np.array(np.uint8(img))
            
            
            self.m['ret']   = ret
            self.m['frame'] = frame
            self.m['index'] = self.index
#                 time.sleep(0.1)
            self.m['p'] += 1

        # stop data acquisition
        cam.stream_off()

        # close device
        cam.close_device()




class Consumer(Process):
    def __init__(self, m, predictor, detector):
        super(Consumer, self).__init__()
        self.m = m
        self.predictor = predictor
        self.detector = detector
        
    def run(self):
        time1=cv2.getTickCount()
        while True:
            if self.m['c'] < self.m['p']:
                iscontinue = 0
                
                ret   = self.m['ret']
                frame = self.m['frame']
                index = self.m['index']
                self.m['c'] += 1
                
                print ('frame index: ', index)
                cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
                cv2.imshow('frame', frame)
                key = cv2.waitKey(10)
                if key == 113:  # q : quit
                    break
        time2=cv2.getTickCount()
        print("each frame consuming time: ",(time2-time1)/cv2.getTickFrequency()/self.m['index'])
        cv2.destroyAllWindows()


if __name__ == '__main__':
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
    p1 = Producer(m,exposure=20000)
    c1 = Consumer(m, predictor, detector)

    p1.start()
    c1.start()
    p1.join()
    c1.join()
    
    
    