import numpy as np
import scipy as sp
from numpy import linalg
from scipy.stats import linregress
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

num = 37
#r = np.zeros(num)

ks, bs, cs = np.zeros(num), np.zeros(num), np.zeros(num)
draw = True
#size = 25
a=np.load('/home/fengguo/microstress/background.npy')
for i in range(num):
    mean, std = np.load('mean%i.npy'%i), np.load('std%i.npy'%i)
    var = std**2
    plt.axis([0, mean.max()+1, 0, var.max()+1])
    k, b, c = linregress(mean.ravel(), var.ravel())[0:3]
    t = np.arange(0, mean.max(), mean.max()/10)
    s = k * t + b
    #r[i] = c
    print c
    if True:
        fig, ax = plt.subplots(1)
        ax.set_xlabel('average of gray level at each pixel',size='xx-large')
        ax.set_ylabel('standard variance of gray level at each pixel',size='xx-large')
        #plt.ylabel('standard variance of gray level at each pixel')
        ax.plot(t, s, 'r')
        ax.plot(mean.ravel(), var.ravel(), '.')
        ax.annotate("",xy=(10,0),xycoords='data',xytext=(10,b),textcoords='data',arrowprops=dict(arrowstyle="<->",patchA=None,shrinkA=0,patchB=None,shrinkB=0))#,connectionstyle="arc3"),)
        ax.plot((0,20),(b,b),'k')
        ax.plot((0,20),(0,0),'k')
        ax.text(10,b/2,'$\sigma^2-\\alpha\mu$',size='xx-large')
        #fig.show()
        fig.savefig('/home/fengguo/Chapter2/mean%i_.pdf'%i,format='pdf', bbox_inches='tight', pad_inches=0)
        plt.clf()
    ks[i], bs[i], cs[i] = k, b, c
"""
i = 7
mean_, var_ = mean.ravel(), var.ravel()
n = mean_.shape[0]
C = np.zeros((2*n,n+2))
b = np.zeros(2*n)
for i in range(n):
    C[2*i, 0] = 1
    C[2*i+1, 1] = 1
    C[2*i, 2+i] = 1
    C[2*i+1, 2+i] = k
    b[2*i] = mean_[i]
    b[2*i+1] = var_[i]
solve = linalg.solve(np.dot(C.T,C),np.dot(C.T,b))
"""

np.save('/home/fengguo/coefficients',cs)
np.save('/home/fengguo/alphas',ks)

fig, ax = plt.subplots(1)
plt.plot(ks, bs, '.')
plt.axis([0, ks.max(), bs.min(), 4])
ax.set_xlabel('$\\alpha$',size = 'xx-large')
ax.set_ylabel('$\sigma^2-\\alpha\mu$',size = 'xx-large')
n_mu, sigsq, r = linregress(ks,bs)[0:3]
ax.annotate("",xy=(0.002,0),xycoords='data',xytext=(0.002,sigsq),textcoords='data',arrowprops=dict(arrowstyle="<->",patchA=None,shrinkA=0,patchB=None,shrinkB=0))
ax.plot((0,0.003),(sigsq,sigsq),'k')
ax.plot((0,0.003),(0,0),'k')
ax.text(0.0025,sigsq/2,'$\sigma^2$',size='xx-large')
t = np.array([0,ks.max()*1.1])
s = n_mu * t + sigsq
ax.plot(t,s)
#plt.show()
print r
plt.savefig('/home/fengguo/Chapter2/overall.pdf',format='pdf', bbox_inches='tight', pad_inches=0)
