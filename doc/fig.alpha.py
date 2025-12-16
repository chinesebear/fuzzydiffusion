import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/times.ttf')
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
import pandas as pd
import numpy as np

fig, axe = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
plt.rcParams.update({'font.size': 18})

alpha=np.arange(0,1.05,0.05)
FID=[
    [4.464,4.384,4.261,4.1904,4.0871,3.9064,3.81,4.075,4.356,5.054,5.31,5.621,5.952,6.233,6.568,6.822,7.000,7.201,7.202,7.203,7.204], ## lsun church 3.81
    [4.429,4.0238,3.6044,3.264,3.195,2.9135,2.81,3.255,4.521,5.656,6.82,7.044,7.568,8.052,8.23,8.386,8.515,8.572,8.538,8.579,8.598,], ## lsun bedroom 2.81
    [14.669,14.102,13.823,13.509,13.134,12.658,12.024,11.436,11.061,10.69,9.584,7.81,7.087,6.461,6.29,7.069,7.537,7.818,7.956,8.200,8.300], ## ms coco 6.29
]

plt.subplot(1,3,1)
plt.plot(alpha, FID[0], color='skyblue', linewidth=3)
plt.scatter([0.3],[3.81], color='r')
plt.plot([0, 0.3], [3.81, 3.81], color='r', linestyle='dotted')
plt.plot([0.3, 0.3], [0, 3.81], color='r', linestyle='dotted')
plt.text(0.3, 3.81, r'$\alpha$=0.312',fontsize=18)
plt.ylim(1,15)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel(r'$\alpha$',fontsize=18)
plt.ylabel('FID',fontsize=18)
plt.title('LSUN Church') 

plt.subplot(1,3,2)
plt.plot(alpha, FID[1], color='skyblue',linewidth=3)
plt.scatter([0.3],[2.81], color='r')
plt.plot([0, 0.3], [2.81, 2.81], color='r', linestyle='dotted')
plt.plot([0.3, 0.3], [0, 2.81], color='r', linestyle='dotted')
plt.text(0.3, 2.81, r'$\alpha$=0.301',fontsize=18)
plt.ylim(1,15)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel(r'$\alpha$',fontsize=18)
plt.ylabel('FID',fontsize=18)
plt.title('LSUN Bedroom')


plt.subplot(1,3,3)
plt.plot(alpha, FID[2], color='skyblue',linewidth=3)
plt.scatter([0.7],[6.29], color='r')
plt.plot([0, 0.7], [6.29, 6.29], color='r', linestyle='dotted')
plt.plot([0.7, 0.7], [0, 6.29], color='r', linestyle='dotted')
plt.text(0.7, 6.29, r'$\alpha$=0.720',fontsize=18)
plt.ylim(1,25)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel(r'$\alpha$',fontsize=18)
plt.ylabel('FID',fontsize=18)
plt.title('MS COCO')



plt.tight_layout()#调整整体空白
plt.savefig("doc/fig.alpha.svg", format = "svg", transparent=True)
plt.savefig("doc/fig.alpha.jpg", format = "jpg", transparent=True,dpi=600)
plt.show()