from picamera import PiCamera
import numpy as np
import matplotlib.pyplot as plt

h = 1024 # change this to anything < 2592 (anything over 2000 will likely get a memory error when plotting
cam_res = (int(h),int(0.75*h)) # keeping the natural 3/4 resolution of the camera
# we need to round to the nearest 16th and 32nd (requirement for picamera)
cam_res = (int(16*np.floor(cam_res[1]/16)),int(32*np.floor(cam_res[0]/32)))
# camera initialization
cam = PiCamera()
cam.resolution = (cam_res[1],cam_res[0])
data = np.empty((cam_res[0],cam_res[1],3),dtype=np.uint8) # preallocate image
while True:
    try:
        cam.capture(data,'rgb') # capture RGB image
        plt.imshow(data) # plot image
        # clear data to save memory and prevent overloading of CPU
        data = np.empty((cam_res[0],cam_res[1],3),dtype=np.uint8)
        plt.show() # show the image
        # press enter when ready to take another photo
        input("Click to save a different plot")
    # pressing CTRL+C exits the loop
    except KeyboardInterrupt:
        break
