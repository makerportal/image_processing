import time
from picamera import PiCamera
import scipy.ndimage as scimg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from sklearn.cluster import DBSCAN

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
t1 = time.time()
gaus = scimg.fourier_gaussian(data[:,:,0],sigma=0.01)
can_x = scimg.prewitt(gaus,axis=0)
can_y = scimg.prewitt(gaus,axis=1)
can = np.hypot(can_x,can_y)

# pulling out object edges
fig3,ax3 = plt.subplots(2,1,figsize=(10,7))
ax3[0].pcolormesh(x,y,can,cmap='gray')
bin_size = 100 # total bins to show
percent_cutoff = 0.02 # cutoff once main peak tapers to 1% of max
hist_vec = np.histogram(can.ravel(),bins=bin_size)
hist_x,hist_y = hist_vec[0],hist_vec[1]
for ii in range(np.argmax(hist_x),bin_size):
    hist_max = hist_y[ii]
    if hist_x[ii]<percent_cutoff*np.max(hist_x):
        break

# sklearn section for clustering
x_cluster = x[can>hist_max]
y_cluster = y[can>hist_max]
scat_pts = []
for ii,jj in zip(x_cluster,y_cluster):
    scat_pts.append((ii,jj))
    
min_samps = 15
leaf_sz = 10
max_dxdy = 25
# clustering analysis for object detection
clustering = DBSCAN(eps=max_dxdy,min_samples=min_samps,
                    algorithm='kd_tree',
                    leaf_size=leaf_sz).fit(scat_pts)

color_txt = ['Red','Green','Blue']
fig4,ax4 = plt.subplots(1)
fig4.set_size_inches(9,7)
im_show = ax4.imshow(data,origin='lower')
# drawing boxes around individual objects
for ii in np.unique(clustering.labels_):
    if ii==-1:
        continue
    clus_dat = np.where(clustering.labels_==ii)

    x_pts = x_cluster[clus_dat]
    y_pts = y_cluster[clus_dat]
    ax3[1].plot(x_pts,y_pts,marker='.',linestyle='',label='Object {0:2.0f}'.format(ii))

ax3[1].legend()
fig3.savefig('dbscan_demo.png',dpi=150,facecolor=[252/255,252/255,252/255])
plt.show()
