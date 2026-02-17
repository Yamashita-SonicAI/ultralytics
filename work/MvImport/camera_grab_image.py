# -- coding: utf-8 --

import os
import sys
import termios
import threading
import time
from ctypes import *

import cv2
import numpy as np

from MvImport.MvCameraControl_class import *
g_bExit = False

class cs_camera:
	def __init__(self):
		self.cam, self.use_camera_name = self.Initialize()
		if "CS42" in self.use_camera_name:
			self.WIDTH, self.HEIGHT = 720, 540
		elif "EG42" in self.use_camera_name:
			self.WIDTH, self.HEIGHT = 720, 540
		elif "CS160" in self.use_camera_name:
			self.WIDTH, self.HEIGHT = 1440, 1080
		elif "CS500U" in self.use_camera_name:
			self.WIDTH, self.HEIGHT = 2600, 2160
		else:
			print("対応していないカメラです")
			sys.exit(1)

	def Initialize(self):

		SDKVersion = MvCamera.MV_CC_GetSDKVersion()
		print ("SDKVersion[0x%x]" % SDKVersion)

		deviceList = MV_CC_DEVICE_INFO_LIST()
		tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
		
		# ch:枚举设备 | en:Enum device
		ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
		if ret != 0:
			print ("enum devices fail! ret[0x%x]" % ret)
			sys.exit()

		if deviceList.nDeviceNum == 0:
			print ("find no device!")
			sys.exit()

		print ("Find %d devices!" % deviceList.nDeviceNum)
		
		usb_index = None
		gige_index = None

		for i in range(0, deviceList.nDeviceNum):
			mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
			if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
				print ("\ngige device: [%d]" % i)
				strModeName = ""
				for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
					strModeName = strModeName + chr(per)
				print ("device model name: %s" % strModeName)

				nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
				nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
				nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
				nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
				print ("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
				gige_index = i
			elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
				print ("\nu3v device: [%d]" % i)
				strModeName = ""
				for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
					if per == 0:
						break
					strModeName = strModeName + chr(per)
				print ("device model name: %s" % strModeName)

				strSerialNumber = ""
				for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
					if per == 0:
						break
					strSerialNumber = strSerialNumber + chr(per)
				print ("user serial number: %s" % strSerialNumber)
				
				usb_index = i

		if usb_index is not None:
			nConnectionNum = usb_index
			print("Automatically selected USB device [%d]" % nConnectionNum)
		else:
			nConnectionNum = gige_index
			print("Automatically selected GigE device [%d]" % nConnectionNum)
		# USB が無い → GigE 用にユーザ入力
			# if sys.version >= '3':
			# 	nConnectionNum = input("please input the number of the device to connect:")
			# else:
			# 	nConnectionNum = raw_input("please input the number of the device to connect:")

		if int(nConnectionNum) >= deviceList.nDeviceNum:
			print ("intput error!")
			sys.exit()

		# ch:创建相机实例 | en:Creat Camera Object
		cam = MvCamera()
		
		# ch:选择设备并创建句柄| en:Select device and create handle
		stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents
		# 使うカメラの製品名を取得
		if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
			strModeName = ""
			for per in stDeviceList.SpecialInfo.stGigEInfo.chModelName:
				strModeName = strModeName + chr(per)

			nip1 = ((stDeviceList.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
			nip2 = ((stDeviceList.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
			nip3 = ((stDeviceList.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
			nip4 = (stDeviceList.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)

		elif stDeviceList.nTLayerType == MV_USB_DEVICE:
			strModeName = ""
			for per in stDeviceList.SpecialInfo.stUsb3VInfo.chModelName:
				if per == 0:
					break
				strModeName = strModeName + chr(per)
		use_camera_name = strModeName
		print(use_camera_name)
		ret = cam.MV_CC_CreateHandle(stDeviceList)
		if ret != 0:
			print ("create handle fail! ret[0x%x]" % ret)
			sys.exit()
		
		return cam, use_camera_name
	
	def destroy_handle(self):
		self.cam.MV_CC_DestroyHandle()

	def ApplySetting(self, setting):
		if "resolution" in setting:
			self.cam.MV_CC_SetIntValue("Width", int(setting["resolution"]*self.WIDTH/4)*4)
			self.cam.MV_CC_SetIntValue("Height", int(setting["resolution"]*self.HEIGHT/4)*4)
		if "exposure_time" in setting:
			self.cam.MV_CC_SetFloatValue("ExposureTime", setting["exposure_time"])
			# cam.MV_CC_SetBoolValue("AcquisitionFrameRateControlEnable", True)
		if "fps" in setting: 
			self.cam.MV_CC_SetFloatValue("AcquisitionFrameRate", setting["fps"])

	def camera_thread(self, stop_event, output_queue=None, exposure_time=4000, fps=15.0, setting_data=None):
		cam = self.cam
		use_camera_name = self.use_camera_name
		width = self.WIDTH
		height = self.HEIGHT

		ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
		if ret != 0:
			print ("open device fail! ret[0x%x]" % ret)
			sys.exit()

		# ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
		if "EG" in use_camera_name:
			nPacketSize = cam.MV_CC_GetOptimalPacketSize()
			if int(nPacketSize) > 0:
				ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize",nPacketSize)
				if ret != 0:
					print ("Warning: Set Packet Size fail! ret[0x%x]" % ret)
			else:
				print ("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)
		

		# ch:设置触发模式为off | en:Set trigger mode as off
		ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
		if ret != 0:
			print ("set trigger mode fail! ret[0x%x]" % ret)
			sys.exit()

		#########
		#設定変更#
		cam.MV_CC_SetIntValue("Width", width)
		cam.MV_CC_SetIntValue("Height", height)
		cam.MV_CC_SetFloatValue("ExposureTime", exposure_time)
		cam.MV_CC_SetFloatValue("AcquisitionFrameRate", fps)
		cam.MV_CC_SetBoolValue("AcquisitionFrameRateControlEnable", True) 
		cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True) 
		
		#########
		#########

		# ch:获取数据包大小 | en:Get payload size
		stParam =  MVCC_INTVALUE()
		memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
		if "CS42" in use_camera_name:
			ret = cam.MV_CC_SetEnumValueByString("PixelFormat", "BGR8Packed")
		elif "EG42" in use_camera_name:
			ret = cam.MV_CC_SetEnumValueByString("PixelFormat", "BGR8")
		elif "CS160" in use_camera_name:
			ret = cam.MV_CC_SetEnumValueByString("PixelFormat", "BayerRG8")
		elif "CS500U" in use_camera_name:
			ret = cam.MV_CC_SetEnumValueByString("PixelFormat", "BayerBG8")
		ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)
		if ret != 0:
			print ("get payload size fail! ret[0x%x]" % ret)
			sys.exit()
		nPayloadSize = stParam.nCurValue

		# ch:开始取流 | en:Start grab image
		ret = cam.MV_CC_StartGrabbing()
		if ret != 0:
			print ("start grabbing fail! ret[0x%x]" % ret)
			sys.exit()

		data_buf = (c_ubyte * nPayloadSize)()
		pData = byref(data_buf)
		stFrameInfo = MV_FRAME_OUT_INFO_EX()
		memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
		print("start camera thread")
		time.sleep(0.5)

		while not stop_event.is_set():
			# stParam =  MVCC_FLOATVALUE()
			# memset(byref(stParam), 0, sizeof(MVCC_FLOATVALUE))
			# ret = cam.MV_CC_GetFloatValue("ResultingFrameRate", stParam)
			ret = cam.MV_CC_GetOneFrameTimeout(pData, nPayloadSize, stFrameInfo, 1000)
			packet = {}
			if ret == 0:
				# print ("get one frame: Width[%d], Height[%d], PixelType[0x%x], nFrameNum[%d]"  % (stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.enPixelType,stFrameInfo.nFrameNum))
				if "CS42" in use_camera_name:
					image = np.asarray(pData._obj)
					image = image.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth, 3))
				elif "EG42" in use_camera_name:
					image = np.asarray(pData._obj)
					image = image.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth, 3))
				elif "CS160" in use_camera_name:
					image = np.asarray(pData._obj)
					image = image.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
					image = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2RGB)
				elif "CS500U" in use_camera_name:
					image = np.asarray(pData._obj)
					image = image.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
					image = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2RGB)
				else:
					image = np.asarray(pData._obj)
					image = image.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth, 3))
				flipped = cv2.flip(image, -1)  # Flip the image horizontally
				raw_flipped = flipped.copy()
				processed_flipped = flipped.copy()
				# フレーム生成時のad_inferenceをパケットに埋め込む
				ad_inference_at_capture = setting_data.get("ad_inference", True) if setting_data else True
				packet = {"raw" : raw_flipped, "processed" : processed_flipped, "ad_inference": ad_inference_at_capture}
				output_queue.put(packet)
				#print(f"Frame {stFrameInfo.nFrameNum} processed and added to queue.")
				# cv2.imshow("show", flipped)
				# cv2.waitKey(1)
			else:
				print ("no data[0x%x]" % ret)
				continue
		
		print ("stop event set")
		output_queue.put(None)

	def stop_camera_thread(self):
		print("stop camera thread")
		self.cam.MV_CC_StopGrabbing()
		self.cam.MV_CC_CloseDevice()
	
	def open_cam(self):
		# デバイスオープン
		cam = self.cam
		ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
		if ret != 0:
			print("Open device fail! ret[0x%x]" % ret)
			sys.exit()
		# ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
		if "EG" in self.use_camera_name:
			nPacketSize = cam.MV_CC_GetOptimalPacketSize()
			if int(nPacketSize) > 0:
				ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize",nPacketSize)
				if ret != 0:
					print ("Warning: Set Packet Size fail! ret[0x%x]" % ret)
			else:
				print ("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)
	
	def close_cam(self):
		# デバイスクローズ
		self.cam.MV_CC_CloseDevice()

	def set_param(self, width, height, exposure_time=10000, fps=30):
		cam = self.cam
		use_camera_name = self.use_camera_name
		print(width, height)

		# if width == 0 or height == 0 :
		# 	width = self.WIDTH
		# 	height = self.HEIGHT
		# トリガーモードOFF
		cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)

		# 画面設定
		cam.MV_CC_SetIntValue("Width", width)
		cam.MV_CC_SetIntValue("Height", height)
		cam.MV_CC_SetFloatValue("ExposureTime", exposure_time)
		cam.MV_CC_SetFloatValue("AcquisitionFrameRate", fps)
		cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
		if "CS42" in use_camera_name:
			cam.MV_CC_SetEnumValueByString("PixelFormat", "BGR8Packed")
		elif "EG42" in use_camera_name:
			cam.MV_CC_SetEnumValueByString("PixelFormat", "BGR8")
		elif "CS160" in use_camera_name:
			cam.MV_CC_SetEnumValueByString("PixelFormat", "BayerRG8")
		elif "CS500U" in use_camera_name:
			cam.MV_CC_SetEnumValueByString("PixelFormat", "BayerBG8")

		# Payloadサイズ取得
		stParam = MVCC_INTVALUE()
		memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
		ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)
		if ret != 0:
			print("Get payload size fail! ret[0x%x]" % ret)
			sys.exit()
		nPayloadSize = stParam.nCurValue
		return nPayloadSize
	
	def start_capture(self):
		cam = self.cam
		# キャプチャ開始
		ret = cam.MV_CC_StartGrabbing()
		if ret != 0:
			print("Start grabbing fail! ret[0x%x]" % ret)
			sys.exit()
	
	def stop_capture(self):
		self.cam.MV_CC_StopGrabbing()
	
	def grab_frame(self, nPayloadSize):
		cam = self.cam
		use_camera_name = self.use_camera_name

		data_buf = (c_ubyte * nPayloadSize)()
		pData = byref(data_buf)
		stFrameInfo = MV_FRAME_OUT_INFO_EX()
		memset(byref(stFrameInfo), 0, sizeof(MV_FRAME_OUT_INFO_EX))

		ret = cam.MV_CC_GetOneFrameTimeout(pData, nPayloadSize, stFrameInfo, 1000)
		if ret == 0:
			if "CS42" in use_camera_name:
				image = np.asarray(pData._obj).reshape((stFrameInfo.nHeight, stFrameInfo.nWidth, 3))
			elif "EG42" in use_camera_name:
				image = np.asarray(pData._obj).reshape((stFrameInfo.nHeight, stFrameInfo.nWidth, 3)) 
			elif "CS160" in use_camera_name:
				image = np.asarray(pData._obj)
				image = image.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
				image = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2RGB)
			elif "CS500U" in use_camera_name:
				image = np.asarray(pData._obj)
				image = image.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
				image = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2RGB)
			else :
				image = np.asarray(pData._obj).reshape((stFrameInfo.nHeight, stFrameInfo.nWidth, 3))
			flipped = cv2.flip(image, -1)
			# flipped = image
			return flipped
		else:
			return None