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

# different edge detection methods
fig,ax = plt.subplots(2,1,figsize=(10,6))
t1 = time.time()
cam.capture(data,'rgb') # capture image

fig2,ax2 = plt.subplots(1,1,figsize=(12,8))
ax2.imshow(data)
 
scale_val = 0.25
min_samps = 20
leaf_sz = 15
max_dxdy = 35
gaus = scimg.fourier_gaussian(scimg.zoom(np.mean(data,2),scale_val),sigma=0.01)
x,y = np.meshgrid(np.arange(0,np.shape(data)[1],1/scale_val),
                  np.arange(0,np.shape(data)[0],1/scale_val))

# Canny method without angle   
can_x = scimg.prewitt(gaus,axis=0)
can_y = scimg.prewitt(gaus,axis=1)
can = np.hypot(can_x,can_y)
ax[0].pcolormesh(x,y,gaus)
# pulling out object edges
bin_size = 100 # total bins to show
percent_cutoff = 0.018 # cutoff once main peak tapers to 1% of max
hist_vec = np.histogram(can.ravel(),bins=bin_size)
hist_x,hist_y = hist_vec[0],hist_vec[1]
for ii in range(np.argmax(hist_x),bin_size):
    hist_max = hist_y[ii]
    if hist_x[ii]<percent_cutoff*np.max(hist_x):
        break

# sklearn section for clustering
x_cluster = x[can>hist_max]
y_cluster = y[can>hist_max]
x_scaled = np.where(can>hist_max,x,0)
y_scaled = np.where(can>hist_max,y,0)
scat_pts = []
for ii,jj in zip(x_cluster,y_cluster):
    scat_pts.append((ii,jj))
    
# clustering analysis for object detection
clustering = DBSCAN(eps=max_dxdy,min_samples=min_samps,
                    algorithm='ball_tree',
                    leaf_size=leaf_sz).fit(scat_pts)
nn_time = time.time()-t1

# looping through each individual object
for ii in np.unique(clustering.labels_):    
    if ii==-1:
        continue
    clus_dat = np.where(clustering.labels_==ii)

    x_pts = x_cluster[clus_dat]
    y_pts = y_cluster[clus_dat]
    cent_mass = (np.mean(x_pts),np.mean(y_pts))
    if cent_mass[0]<np.min(x)+10 or cent_mass[0]>np.max(x)-10 or\
       cent_mass[1]<np.min(y)+10 or cent_mass[1]>np.max(y)-10:
        continue
    
    ax[1].plot(x_pts,y_pts,marker='.',linestyle='',
               label='Unrotated Scatter')
    # rotation algorithm
    evals,evecs = np.linalg.eigh(np.cov(x_pts,y_pts))
    angle = np.arctan(evecs[0][1]/evecs[0][0])
    print(str((angle/np.pi)*180))
    rot_vec = np.matmul(evecs.T,[x_pts,y_pts])
    
    # rectangle algorithms
    if angle<0:
        rect_origin = (np.matmul(evecs,[np.min(rot_vec[0]),np.max(rot_vec[1])]))
    else:
        rect_origin = (np.matmul(evecs,[np.max(rot_vec[0]),np.min(rot_vec[1])]))
        
    rect_width = np.max(rot_vec[0])-np.min(rot_vec[0])
    rect_height = np.max(rot_vec[1])-np.min(rot_vec[1])
    obj_rect = Rectangle(rect_origin,rect_width,rect_height,angle=(angle/np.pi)*180)
    pc = PatchCollection([obj_rect],facecolor='',edgecolor='r',linewidth=2)
    ax2.add_collection(pc)
    ax2.annotate('{0:2.0f}$^\circ$ Rotation'.format((angle/np.pi)*180),
                 xy=(rect_origin),
                 xytext=(0,0),textcoords='offset points',
                 bbox=dict(fc='white'))
    
ax[1].set_xlim(np.min(x),np.max(x))
ax[1].set_ylim(np.min(y),np.max(y))

print('total time: {0:2.1f}'.format(time.time()-t1))
fig2.savefig('rectangles_over_real_image.png',dpi=200,facecolor=[252/255,252/255,252/255])
plt.show()
