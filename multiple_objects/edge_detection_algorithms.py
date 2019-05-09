import time
from picamera import PiCamera
import scipy.ndimage as scimg
import numpy as np
import matplotlib.pyplot as plt

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
# diff of gaussians
t0 = time.time()
grad_xy = scimg.gaussian_gradient_magnitude(data[:,:,0],sigma=1.5)
##grad_xy = np.mean(grad_xy,2)
t_grad_xy = time.time()-t0
# laplacian of gaussian
t0 = time.time()
lap = scimg.gaussian_laplace(data[:,:,0],sigma=0.7)
t_lap = time.time()-t0
# Canny method without angle
t0 = time.time()
gaus = scimg.fourier_gaussian(data[:,:,0],sigma=0.05)
can_x = scimg.prewitt(gaus,axis=0)
can_y = scimg.prewitt(gaus,axis=1)
can = np.hypot(can_x,can_y)
##can = np.mean(can,2)
t_can = time.time()-t0
# Sobel method
t0 = time.time()
sob_x = scimg.sobel(data[:,:,0],axis=0)
sob_y = scimg.sobel(data[:,:,0],axis=1)
sob = np.hypot(sob_x,sob_y)
##sob = np.mean(sob,2)
t_sob = time.time()-t0

# plotting routines and labeling
fig,ax = plt.subplots(2,2,figsize=(12,6))
ax[0,0].pcolormesh(x,y,grad_xy,cmap='gray')
ax[0,0].set_title(r'Gaussian Gradient [$\sigma = 1.5$] (Computation Time: {0:2.2f}s)'.format(t_grad_xy))
ax[0,1].pcolormesh(x,y,lap,cmap='gray')
ax[0,1].set_title(r'Laplacian of Gaussian [$\sigma = 0.7$] (Computation Time: {0:2.2f}s)'.format(t_lap))
ax[1,0].pcolormesh(x,y,can,cmap='gray')
ax[1,0].set_title(r'Canny [$\sigma = 0.05$] (Computation Time: {0:2.2f}s)'.format(t_can))
ax[1,1].pcolormesh(x,y,sob,cmap='gray')
ax[1,1].set_title('Sobel (Computation Time: {0:2.2f}s)'.format(t_sob))
fig.tight_layout()
fig.savefig('edge_plots.png',dpi=150,facecolor=[252/255,252/255,252/255])
#analyzing histograms
fig2,ax2 = plt.subplots(2,2,figsize=(12,6))
ax2[0,0].hist(grad_xy.ravel(),bins=100)
ax2[0,1].hist(lap.ravel(),bins=100)
ax2[1,0].hist(can.ravel(),bins=100)
ax2[1,1].hist(sob.ravel(),bins=100)

# pulling out object edges
fig3,ax3 = plt.subplots(3,1,figsize=(12,6))
ax3[0].pcolormesh(x,y,can)
bin_size = 100
hist_vec = ax3[1].hist(can.ravel(),bins=bin_size)
hist_x,hist_y = hist_vec[0],hist_vec[1]
for ii in range(np.argmax(hist_x),bin_size):
    hist_max = hist_y[ii]
    if hist_x[ii]<0.01*np.max(hist_x):
        break
    
ax3[2].plot(x[can>hist_max],y[can>hist_max],marker='.',linestyle='')

plt.show()
