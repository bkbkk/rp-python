import gxipy as gx
import sys
import time
from PIL import Image
import cv2
import numpy as np
import threading

class GxiCam():
    def __init__(self,exposure=2000,width=1280,
                 width_offset=0,height=1024,height_offset=0):
        self.exposure       = exposure
        self.width          = width
        self.height         = height
        self.width_offset   = width_offset if not width_offset == 0 else None
        self.height_offset  = height_offset if not height_offset == 0 else None
        self.cam = None
        self.data_stream = None
    
    def open(self):
        device_manager = gx.DeviceManager()
        dev_num, dev_info_list = device_manager.update_device_list()
        if dev_num == 0:
            print("Number of enumerated devices is 0")
            return
        # open the first device by exclusive mode
        self.cam ,self.handle= device_manager.open_device_by_index(1, 4)
        print (self.handle)
        #设置缓冲区大小
        cam_datastream=gx.DataStream(self.handle)
        cam_datastream.set_acquisition_buffer_number(buf_num=1)

        # 设置参数
        # exit when the camera is a mono camera
        if self.cam.PixelColorFilter.is_implemented() is False:
            print("This sample does not support mono camera.")
            self.cam.close_device()
            return
        # set continuous acquisition
        if self.cam.AcquisitionMode.is_implemented() is True:
            print("the camera does support continue GxAcquisitionMode")
            self.cam.AcquisitionMode.set(gx.GxAcquisitionModeEntry.CONTINUOUS)
        else:
            print("the camera does not support continue GxAcquisitionMode")
        self.set_param()
        self.get_info()
    
    
    def start(self):
        self.data_stream=self.cam.data_stream[0]
        # start data acquisition
        self.cam.stream_on()
        
    
    def get_info(self):
        print("相机的属性 ", 
              " \n width:       ", self.cam.Width.get(), 
              " \n Height:      ", self.cam.Height.get(),
              " \n exposure:    ", self.cam.ExposureTime.get(), 
              " \n TriggerMode: ", self.cam.TriggerMode.get())
    
    
    def close(self):
        # stop data acquisition
        self.cam.stream_off()
        # close device
        self.cam.close_device()
    
    
    def set_param(self):
        if self.cam is None:
            print ('the gxi cam not open yet')
            return
        # set exposure
        self.cam.ExposureTime.set(self.exposure)
        # set width
        self.cam.Width.set(self.width)
        self.cam.Height.set(self.height)
        # offset 
        if not self.width_offset is None:
            self.cam.OffsetX.set(self.width_offset)
        if not self.height_offset is None:
            self.cam.OffsetY.set(self.height_offset)
        # set trigger mode and trigger source
        self.cam.TriggerMode.set(gx.GxSwitchEntry.OFF)
        self.cam.TriggerSource.set(gx.GxTriggerSourceEntry.SOFTWARE)
        
    
    def get_img(self):
        if self.data_stream is None:
            print ('data stream not init')
            return
        self.data_stream.flush_queue()
        ret=True
        raw_image = self.data_stream.get_image()
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
        img = Image.fromarray(numpy_image, 'RGB')
        frame=np.array(np.uint8(img))
        
        return ret, frame
    
    
    
    
if __name__ == '__main__':
    cam = GxiCam()
    cam.open()
    cam.start()
    
    while True:
        ret, frame = cam.get_img()
        if not ret:
            print ('ret: ', ret)
            break
            
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
        
    cam.close()