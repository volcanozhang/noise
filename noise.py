import numpy as np
import numpy.ma as ma
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pylab

from math import pow, sqrt, exp, log, sin, cos, pi
from common import stringint

xy = np.load('peaks.npy')

offset = 4096
nimage = 100
framedim = (2594, 2774)
images = np.zeros((100, framedim[0], framedim[1]))
nb_elem = framedim[0] * framedim[1]
formatdata = np.uint16
dirpath = '/media/CASTEL_1TO/BM32_Feb_2013/21Feb13/Si1g_200N/nomvt/'

bkg = np.load("/home/fengguo/microstress/background.npy")
images = np.zeros((100,21,21))
mean, std = np.zeros((21,21)), np.zeros((21,21))
"""
for i in range(xy.shape[0]):
    xi, yi = int(round(xy[i,1])), int(round(xy[i,0]))
    for j in range(100):
        path = dirpath + 'S1gnomvt_%s_mar.tiff'%stringint(j, 4)
        f = open(path, 'rb')
        f.seek(offset)
        images[j] = np.fromfile(f, dtype = formatdata, count = nb_elem).reshape(framedim)[xi-10: xi+11, yi-10: yi+11]+bkg[xi-10: xi+11, yi-10: yi+11]-100
        f.close()
    for k in range(21):
        for l in range(21):
            mean[k, l], std[k, l] = images[:, k, l].mean(), images[:, k, l].std()
    #plt.imshow(mean)
    #plt.savefig('mean%i.png'%i,format='png', bbox_inches='tight', pad_inches=0)
    np.save('mean%i'%i,mean)
    np.save('std%i'%i,std)
"""
size = 7
size/2
thre = 110
pk_bound = np.zeros((xy.shape[0],6),np.uint)
for i in range(xy.shape[0]):
    xi, yi = int(round(xy[i,1])), int(round(xy[i,0]))
    path = dirpath + 'S1gnomvt_0000_mar.tiff'
    f = open(path, 'rb')
    f.seek(offset)
    image = np.fromfile(f, dtype = formatdata, count = nb_elem).reshape(framedim)
    xs, ys, xe, ye = xi-size/2, yi-size/2, xi+size/2+1, yi+size/2+1
    xsbl, ysbl, xebl, yebl = True, True, True, True
    while xsbl or ysbl or xebl or yebl:
        xsbl = image[xs-1, ys:ye].max() > thre
        ysbl = image[xs:xe, ys-1].max() > thre
        xebl = image[xe+1, ys:ye].max() > thre
        yebl = image[xs:xe, ye+1].max() > thre
        if xsbl:
            xs = xs-1
        if ysbl:
            ys = ys-1
        if xebl:
            xe = xe+1
        if yebl:
            ye = ye+1
    pk_bound[i] = xi, yi, xs, ys, xe, ye
    mean, std = np.zeros((xe-xs,ye-ys)), np.zeros((xe-xs,ye-ys))
    images = np.zeros((100,xe-xs,ye-ys))
    for j in range(100):
        path = dirpath + 'S1gnomvt_%s_mar.tiff'%stringint(j, 4)
        f = open(path, 'rb')
        f.seek(offset)
        images[j] = np.fromfile(f, dtype = formatdata, count = nb_elem).reshape(framedim)[xs:xe, ys:ye]+bkg[xs:xe, ys:ye]-100
        f.close()
        for k in range(xe-xs):
            for l in range(ye-ys):
                mean[k, l], std[k, l] = images[:, k, l].mean(), images[:, k, l].std()
        np.save('mean%i'%i,mean)
        np.save('std%i'%i,std)
np.save('peak_bound',pk_bound)
    
