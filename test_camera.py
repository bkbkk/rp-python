import gxipy as gx
from PIL import Image
import cv2
import numpy as np
import datetime



if __name__ == '__main__':
    exposure=5000
    width=640
    width_offset=320
    height=1024
    height_offset=0
    
    
    # create a device manager
    device_manager = gx.DeviceManager()
    dev_num, dev_info_list = device_manager.update_device_list()
    if dev_num == 0:
        print("Number of enumerated devices is 0")
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
    # set continuous acquisition
    if cam.AcquisitionMode.is_implemented() is True:
        print("the camera does support continue GxAcquisitionMode")
        cam.AcquisitionMode.set(gx.GxAcquisitionModeEntry.CONTINUOUS)
    else:
        print("the camera does not support continue GxAcquisitionMode")
    # set exposure
    cam.ExposureTime.set(exposure)
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
    cam.Width.set(width)
    cam.Height.set(height)
    # offset Y
    cam.OffsetY.set(height_offset)
    cam.OffsetX.set(width_offset)
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
        data_stream.flush_queue()
        raw_image = data_stream.get_image()
        ret=True
        
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
        
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if not key == -1:
            print (key)
            if key ==113: # q
                cam.Gain.set(1)
            if key == 119:
                cam.Gain.set(16)

    # stop data acquisition
    cam.stream_off()
    # close device
    cam.close_device()


    