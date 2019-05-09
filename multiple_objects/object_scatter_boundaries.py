import time
from picamera import PiCamera
import scipy.ndimage as scimg
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# picamera setup
h = 640 #largest resolution length
cam_res = (int(h),int(0.75*h)) # resizing to picamera's required ratios
cam_res = (int(32*np.floor(cam_res[0]/32)),int(16*np.floor(cam_res[1]/16)))
cam = PiCamera(resolution=cam_res)
# preallocating image variables
data = np.empty((cam_res[1],cam_res[0],3),dtype=np.uint8)
x,y = np.meshgrid(np.arange(cam_res[0]),np.arange(cam_res[1]))

# different edge detection methods
cam.capture(data,'rgb') # capture image

# Canny method without angle
t0 = time.time()
gaus = scimg.fourier_gaussian(data[:,:,0],sigma=0.01)
can_x = scimg.prewitt(gaus,axis=0)
can_y = scimg.prewitt(gaus,axis=1)
can = np.hypot(can_x,can_y)

t_can = time.time()-t0

# pulling out object edges
fig3,ax3 = plt.subplots(2,1,figsize=(10,7))
ax3[0].pcolormesh(x,y,can,cmap='gray')
bin_size = 100 # total bins to show
percent_cutoff = 0.01 # cutoff once main peak tapers to 1% of max
hist_vec = np.histogram(can.ravel(),bins=bin_size)
hist_x,hist_y = hist_vec[0],hist_vec[1]
for ii in range(np.argmax(hist_x),bin_size):
    hist_max = hist_y[ii]
    if hist_x[ii]<percent_cutoff*np.max(hist_x):
        break
    
# scatter points where objects exist
ax3[1].plot(x[can>hist_max],y[can>hist_max],marker='.',linestyle='',
            label='Scatter Above 1% Dropoff')
ax3[1].set_xlim(np.min(x),np.max(x))
ax3[1].set_ylim(np.min(y),np.max(y))
ax3[1].legend()
plt.show()
