import serial 
import os
import struct

CRC8_INIT = 0xff
CRC8_TAB = [
	0x00, 0x5e, 0xbc, 0xe2, 0x61, 0x3f, 0xdd, 0x83, 0xc2, 0x9c, 0x7e, 0x20, 0xa3, 0xfd, 0x1f, 0x41,
	0x9d, 0xc3, 0x21, 0x7f, 0xfc, 0xa2, 0x40, 0x1e, 0x5f, 0x01, 0xe3, 0xbd, 0x3e, 0x60, 0x82, 0xdc,
	0x23, 0x7d, 0x9f, 0xc1, 0x42, 0x1c, 0xfe, 0xa0, 0xe1, 0xbf, 0x5d, 0x03, 0x80, 0xde, 0x3c, 0x62,
	0xbe, 0xe0, 0x02, 0x5c, 0xdf, 0x81, 0x63, 0x3d, 0x7c, 0x22, 0xc0, 0x9e, 0x1d, 0x43, 0xa1, 0xff,
    0x46, 0x18, 0xfa, 0xa4, 0x27, 0x79, 0x9b, 0xc5, 0x84, 0xda, 0x38, 0x66, 0xe5, 0xbb, 0x59, 0x07,
	0xdb, 0x85, 0x67, 0x39, 0xba, 0xe4, 0x06, 0x58, 0x19, 0x47, 0xa5, 0xfb, 0x78, 0x26, 0xc4, 0x9a,
	0x65, 0x3b, 0xd9, 0x87, 0x04, 0x5a, 0xb8, 0xe6, 0xa7, 0xf9, 0x1b, 0x45, 0xc6, 0x98, 0x7a, 0x24,
	0xf8, 0xa6, 0x44, 0x1a, 0x99, 0xc7, 0x25, 0x7b, 0x3a, 0x64, 0x86, 0xd8, 0x5b, 0x05, 0xe7, 0xb9,
	0x8c, 0xd2, 0x30, 0x6e, 0xed, 0xb3, 0x51, 0x0f, 0x4e, 0x10, 0xf2, 0xac, 0x2f, 0x71, 0x93, 0xcd,
	0x11, 0x4f, 0xad, 0xf3, 0x70, 0x2e, 0xcc, 0x92, 0xd3, 0x8d, 0x6f, 0x31, 0xb2, 0xec, 0x0e, 0x50,
	0xaf, 0xf1, 0x13, 0x4d, 0xce, 0x90, 0x72, 0x2c, 0x6d, 0x33, 0xd1, 0x8f, 0x0c, 0x52, 0xb0, 0xee,
	0x32, 0x6c, 0x8e, 0xd0, 0x53, 0x0d, 0xef, 0xb1, 0xf0, 0xae, 0x4c, 0x12, 0x91, 0xcf, 0x2d, 0x73,
	0xca, 0x94, 0x76, 0x28, 0xab, 0xf5, 0x17, 0x49, 0x08, 0x56, 0xb4, 0xea, 0x69, 0x37, 0xd5, 0x8b,
    0x57, 0x09, 0xeb, 0xb5, 0x36, 0x68, 0x8a, 0xd4, 0x95, 0xcb, 0x29, 0x77, 0xf4, 0xaa, 0x48, 0x16,
	0xe9, 0xb7, 0x55, 0x0b, 0x88, 0xd6, 0x34, 0x6a, 0x2b, 0x75, 0x97, 0xc9, 0x4a, 0x14, 0xf6, 0xa8,
	0x74, 0x2a, 0xc8, 0x96, 0x15, 0x4b, 0xa9, 0xf7, 0xb6, 0xe8, 0x0a, 0x54, 0xd7, 0x89, 0x6b, 0x35,
]

CRC_INIT = 0xffff
wCRC_Table = [
	0x0000, 0x1189, 0x2312, 0x329b, 0x4624, 0x57ad, 0x6536, 0x74bf,
	0x8c48, 0x9dc1, 0xaf5a, 0xbed3, 0xca6c, 0xdbe5, 0xe97e, 0xf8f7,
	0x1081, 0x0108, 0x3393, 0x221a, 0x56a5, 0x472c, 0x75b7, 0x643e,
	0x9cc9, 0x8d40, 0xbfdb, 0xae52, 0xdaed, 0xcb64, 0xf9ff, 0xe876,
	0x2102, 0x308b, 0x0210, 0x1399, 0x6726, 0x76af, 0x4434, 0x55bd,
	0xad4a, 0xbcc3, 0x8e58, 0x9fd1, 0xeb6e, 0xfae7, 0xc87c, 0xd9f5,
	0x3183, 0x200a, 0x1291, 0x0318, 0x77a7, 0x662e, 0x54b5, 0x453c,
	0xbdcb, 0xac42, 0x9ed9, 0x8f50, 0xfbef, 0xea66, 0xd8fd, 0xc974,
	0x4204, 0x538d, 0x6116, 0x709f, 0x0420, 0x15a9, 0x2732, 0x36bb,
	0xce4c, 0xdfc5, 0xed5e, 0xfcd7, 0x8868, 0x99e1, 0xab7a, 0xbaf3,
	0x5285, 0x430c, 0x7197, 0x601e, 0x14a1, 0x0528, 0x37b3, 0x263a,
	0xdecd, 0xcf44, 0xfddf, 0xec56, 0x98e9, 0x8960, 0xbbfb, 0xaa72,
	0x6306, 0x728f, 0x4014, 0x519d, 0x2522, 0x34ab, 0x0630, 0x17b9,
	0xef4e, 0xfec7, 0xcc5c, 0xddd5, 0xa96a, 0xb8e3, 0x8a78, 0x9bf1,
	0x7387, 0x620e, 0x5095, 0x411c, 0x35a3, 0x242a, 0x16b1, 0x0738,
	0xffcf, 0xee46, 0xdcdd, 0xcd54, 0xb9eb, 0xa862, 0x9af9, 0x8b70,
	0x8408, 0x9581, 0xa71a, 0xb693, 0xc22c, 0xd3a5, 0xe13e, 0xf0b7,
	0x0840, 0x19c9, 0x2b52, 0x3adb, 0x4e64, 0x5fed, 0x6d76, 0x7cff,
	0x9489, 0x8500, 0xb79b, 0xa612, 0xd2ad, 0xc324, 0xf1bf, 0xe036,
	0x18c1, 0x0948, 0x3bd3, 0x2a5a, 0x5ee5, 0x4f6c, 0x7df7, 0x6c7e,
	0xa50a, 0xb483, 0x8618, 0x9791, 0xe32e, 0xf2a7, 0xc03c, 0xd1b5,
	0x2942, 0x38cb, 0x0a50, 0x1bd9, 0x6f66, 0x7eef, 0x4c74, 0x5dfd,
	0xb58b, 0xa402, 0x9699, 0x8710, 0xf3af, 0xe226, 0xd0bd, 0xc134,
	0x39c3, 0x284a, 0x1ad1, 0x0b58, 0x7fe7, 0x6e6e, 0x5cf5, 0x4d7c,
	0xc60c, 0xd785, 0xe51e, 0xf497, 0x8028, 0x91a1, 0xa33a, 0xb2b3,
	0x4a44, 0x5bcd, 0x6956, 0x78df, 0x0c60, 0x1de9, 0x2f72, 0x3efb,
	0xd68d, 0xc704, 0xf59f, 0xe416, 0x90a9, 0x8120, 0xb3bb, 0xa232,
	0x5ac5, 0x4b4c, 0x79d7, 0x685e, 0x1ce1, 0x0d68, 0x3ff3, 0x2e7a,
	0xe70e, 0xf687, 0xc41c, 0xd595, 0xa12a, 0xb0a3, 0x8238, 0x93b1,
	0x6b46, 0x7acf, 0x4854, 0x59dd, 0x2d62, 0x3ceb, 0x0e70, 0x1ff9,
	0xf78f, 0xe606, 0xd49d, 0xc514, 0xb1ab, 0xa022, 0x92b9, 0x8330,
	0x7bc7, 0x6a4e, 0x58d5, 0x495c, 0x3de3, 0x2c6a, 0x1ef1, 0x0f78
]


class CRC():
    def __init__(self):
        global CRC8_INIT, CRC8_TAB, CRC_INIT, wCRC_Table
        self.CRC8_INIT   = CRC8_INIT
        self.CRC8_TAB    = CRC8_TAB
        self.CRC_INIT    = CRC_INIT
        self.wCRC_Table  = wCRC_Table
    
    def Get_CRC_8(self, data, length, ucCRC8):
        for i in range(length):
            index = ucCRC8 ^ data[i]
            ucCRC8 = self.CRC8_TAB[index]
        return ucCRC8
    
    def Verify_CRC_8(self, data, length):
        expected = 0
        if (length <= 2) or (len(data) <= 2):
            return False
        expected = self.Get_CRC_8(data, length-1, self.CRC8_INIT)
        return expected==data[length-1]
    
    def Append_CRC_8(self, data, length):
        ucCRC = 0
        if (len(data) == 0) or (length <= 2):
            print ('send data head length false')
        ucCRC = self.Get_CRC_8(data, length-1, self.CRC8_INIT)
        data[length-1] = ucCRC
        return data
    
    def Get_CRC_16(self, data, length, wCRC):
        if len(data)==0:
            return 0xffff
        for i in range(length):
            chData = data[i]
            wCRC = wCRC>>8 ^ self.wCRC_Table[(wCRC ^ chData) & 0x00ff]
        return wCRC
    
    def Verify_CRC_16(self, data, length):
        expected = 0
        if (data is None) or (length <= 2):
            return False
        expected = self.Get_CRC_16(data, length-2, self.CRC_INIT)
        return (expected&0xff)==data[length-2] and ((expected>>8)&0xff)==data[length-1]
    
    def Append_CRC_16(self, data, length):
        wCRC= 0
        if (data is None) or (length <= 2):
            print ('send data tail length false')
        wCRC = self.Get_CRC_16(data, length-2, self.CRC_INIT)
        data[length-2] = wCRC & 0x00ff
        data[length-1] = (wCRC >> 8) & 0x00ff
        return data
    
    

    
    
    
class SerialPort():
    def __init__(self, port='/dev/ttyUSB1', bps=115200, timeout=0, writetimeout=0):
        self.port = port
        self.CRC = CRC()
        self.serial = serial.Serial(port, bps, timeout=timeout, writeTimeout=writetimeout)
            
        if not self.serial.isOpen():
            raise Exception('can not open serial port!')
            
        self.rData = {
            'len'        : 0,
            'data_list'  : None,
            'fmt'        : '',
            'fmt_list'   : [],
            'format'     : [
                0xa5,  #head
                int,   #mode
                0x00,  #check
                int,
                float,
                0x00,  #check
                0x00   #check
            ],
            'fmt_list': []
        }
        self.sData = {
            'len'        : 0,
            'data_byte'  : None,
            'fmt'        : '',
            'fmt_list'   : [],
            'format'     : [
                0xa5,
                int,
                0x00,
                float,
                float,
                float,
                int,
                int,
                int,
                int,
                int,
                int,
                0x00,
                0x00
            ]
        }
        self.InitData()
        
        
    def GetParam(self):
        if self.serial.isOpen():
            print('name:         ', self.serial.name) #设备名字
            print('port:         ', self.serial.port)#读或者写端口
            print('baudrate:     ', self.serial.baudrate)#波特率
            print('bytesize:     ', self.serial.bytesize)#字节大小
            print('parity:       ', self.serial.parity)#校验位
            print('stopbits:     ', self.serial.stopbits)#停止位
            print('readTimeout:  ', self.serial.timeout)#读超时设置
            print('writeTimeout: ', self.serial.writeTimeout)#写超时
        else:
            print ('serial not open, can not read param')

    def SetParam(self, bps=115200, readtimeout=0, writetimeout=0):
        self.serial.close()
        self.serial.baudrate = bps
        self.serial.timeout = readtimeout
        self.serial.writeTimeout = writetimeout
        self.serial.open()
        self.GetParam()
        print ('reopen the serial and set the param, now status: ', self.serial.isOpen())
    
    def PrintBuffer(self):
        if self.serial.isOpen():
            print('send    buffer:         ', self.serial.out_waiting) ##获取输入缓冲区的剩余字节数
            print('receive buffer:         ', self.serial.in_waiting)##获取输出缓冲区的字节数
        else:
            print ('serial not open, can not print buffer')
    
    def Send(self, Data):
        self.serial.flushOutput()
#         print ('before out buffer: ', self.serial.out_waiting)
        
        data_byte = self.sParse(Data)
        self.sData['data_byte'] = data_byte
        result = self.serial.write(data_byte)
        
#         print ('after out buffer: ', self.serial.out_waiting)
        return data_byte
        
    
    def Receive(self):
        data_list = []
        if self.serial.in_waiting < self.rData['len']:
            data_list = self.rData['data_list']
        else:
            data = self.serial.read(self.rData['len'])
            if not (self.CRC.Verify_CRC_8(data, 3) and self.CRC.Verify_CRC_16(data, len(data))):
                print ('CRC check fail')
                
            offset = 0
            for fmt in self.rData['fmt_list']:
                if fmt == 'B':
                    data_list.append(struct.unpack(fmt, data[offset:offset+1])[0])
                    offset += 1
                elif fmt == 'f':
                    data_list.append(struct.unpack(fmt, data[offset:offset+4])[0])
                    offset += 4
                else:
                    print ('unsupport fmt now: ', fmt)
            self.rData['data_list'] = data_list
            
        self.serial.flushInput()
        return data_list
    
    
    def Open(self):
        if self.serial.isOpen():
            print ('serial is opened before, dont need to open again')
        else:
            self.serial.open()
        print ('serial open status: ', self.serial.is_open)
    
    def Close(self):
        self.serial.close()
        print ('serial open status: ', self.serial.is_open)
        
    
    def InitData(self):
        for r in self.rData['format']:
            if r == float:
                self.rData['len'] += 4
                self.rData['fmt'] += 'f'
                self.rData['fmt_list'].append('f')
            else:
                self.rData['len'] += 1
                self.rData['fmt'] += 'B'
                self.rData['fmt_list'].append('B')
        for r in self.sData['format']:
            if r == float:
                self.sData['len'] += 4
                self.sData['fmt'] += 'f'
                self.sData['fmt_list'].append('f')
            else:
                self.sData['len'] += 1
                self.sData['fmt'] += 'B'
                self.sData['fmt_list'].append('B')

        print ('receive data length: ', self.rData['len'])
        print ('send    data length: ', self.sData['len'])
        
    def sParse(self, Data):
        mode   = Data['mode']
        pitch  = Data['pitch']
        yaw    = Data['yaw']
        dis    = Data['distance']
        test_1 = Data['test_1']
        test_2 = Data['test_2']
        test_3 = Data['test_3']
        test_4 = Data['test_4']
        test_5 = Data['test_5']
        test_6 = Data['test_6']
        B1 = struct.pack('B', mode)
        f1 = struct.pack('f', pitch)
        f2 = struct.pack('f', yaw)
        f3 = struct.pack('f', dis)
        B2 = struct.pack('B', test_1)
        B3 = struct.pack('B', test_2)
        B4 = struct.pack('B', test_3)
        B5 = struct.pack('B', test_4)
        B6 = struct.pack('B', test_5)
        B7 = struct.pack('B', test_6)
        
        head = b'\xa5' + B1 + b'\x00'
        body = f1 + f2 + f3 + B2 + B3 + B4 + B5 + B6 + B7
        tail = b'\x00\x00'
        data = head + body + tail
        
        data_list = [i for i in data]
        data_list = self.CRC.Append_CRC_8(data_list, 3)
        data_list = self.CRC.Append_CRC_16(data_list, self.sData['len'])
        CRC_check_1 = struct.pack('B', data_list[2])
        CRC_check_2 = struct.pack('B', data_list[-2])
        CRC_check_3 = struct.pack('B', data_list[-1])
        data_byte = b'\xa5' + B1 + CRC_check_1 + body + CRC_check_2 + CRC_check_3
        
        return data_byte
