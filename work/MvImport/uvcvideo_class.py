import cv2
import time

class uvcvideo:
	def __init__(self):
		self.cap = self.Initialize()
		self.width, self.height = 640, 480
		self.fps = 30.0  # デフォルトFPS

	def Initialize(self):
		print("init")
		# V4L2バックエンドを明示的に指定してGStreamer警告を回避
		cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
		if not cap.isOpened():
			print("デバイスが開けません")
			return None

		return cap
	
	def camera_thread(self, stop_event, output_queue=None, exposure_time=10000, fps=30.0):
		if self.cap is None:
			print("カメラが初期化されていません")
			if output_queue:
				output_queue.put(None)
			return

		cap = self.cap
		width, height = self.width, self.height

		cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
		cap.set(cv2.CAP_PROP_FPS, fps)
		self.fps = fps  # 初期FPSを保存

		while not stop_event.is_set():
			frame_start = time.time()

			ret, frame = self.cap.read()
			if ret:
				packet = {"raw" : frame, "processed" : frame}
				output_queue.put(packet)

				# FPSに基づいて待機時間を計算
				target_interval = 1.0 / self.fps
				elapsed = time.time() - frame_start
				sleep_time = target_interval - elapsed
				if sleep_time > 0:
					time.sleep(sleep_time)
			else:
				print ("no data[0x%x]" % ret)
				continue

		print ("stop event set")
		output_queue.put(None)

	def stop_camera_thread(self):
		print("stop camera thread")

	def ApplySetting(self, setting):
		if self.cap is None:
			print("カメラが初期化されていません")
			return
		if "fps" in setting:
			fps_value = setting["fps"]
			# インスタンス変数を更新（camera_threadで使用される）
			self.fps = fps_value
			print(f"ApplySetting: FPS changed to {fps_value}")

	def release(self):
		if self.cap is not None:
			self.cap.release()