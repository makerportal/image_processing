from time import sleep
from picamera import PiCamera

camera = PiCamera()
camera.resolution = (2560,1936)
camera.start_preview()
sleep(5)
camera.capture('test.jpg')
camera.stop_preview()
