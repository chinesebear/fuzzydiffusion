import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/times.ttf')
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
import pandas as pd
import numpy as np

# 生成四个子图 两行两列
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
plt.rcParams.update({'font.size': 18})

# 第一个子图是柱状图，消融，悬空的，四个柱子
x = ['Random', '10%', '50%', '100%']
y = np.array([28.21, 29.45, 34.81, 39.15])
bottom = np.array([0.0, y[0], y[1], y[2]])
delta= y-bottom
errors= np.array([0.51, 0.30, 0.51, 0.62])
bar_labels =[f'+{v:.2f}' for v in (delta)]
bar_labels[0] = f'{y[0]:.2f}'

bars = axs[0, 0].bar(x,delta, color='#4e79a7',edgecolor='black', alpha=0.8, yerr=errors, capsize=5, bottom=bottom)
axs[0, 0].bar_label(bars, labels=bar_labels, padding=3, fontsize=18)
axs[0, 0].tick_params(axis='both', labelsize=18)
axs[0, 0].set_ylim(25, 45)
axs[0, 0].set_ylabel('IS', fontsize=18)
axs[0, 0].set_xlabel('K-Medoids\n(a)', fontsize=18)

# 第二个子图是柱状图，消融，悬空的，三个柱子
x = ['Basic', '+BERT', '+VGG']
y = np.array([30.43, 34.01, 39.15])
bottom = np.array([0.0, y[0], y[1]])
delta= y-bottom 
errors= np.array([0.51, 0.21, 0.51])
bar_labels =[f'+{v:.2f}' for v in (delta)]
bar_labels[0] = f'{y[0]:.2f}'

bars = axs[0, 1].bar(x,delta, color='#4e79a7',edgecolor='black', alpha=0.8, yerr=errors, capsize=5, bottom=bottom)
axs[0, 1].bar_label(bars, labels=bar_labels, padding=3, fontsize=18)
axs[0, 1].tick_params(axis='both', labelsize=18)
axs[0, 1].set_ylim(25, 45)
axs[0, 1].set_ylabel('IS', fontsize=18)
axs[0, 1].set_xlabel('SMC\n(b)', fontsize=18)

# 第三个子图
x = ['Church', 'Bedroom', 'MS COCO']
y = np.array([0.4517, 0.5111, 0.4832])
y2 = np.array([0.4025, 0.4750, 0.4059])
delta= y-y2
bar_labels =[f'+{v:.4f}' for v in (delta)]
# bar_labels[0] = f'{y[0]:.2f}'

bars = axs[1, 0].bar(x,y, color='#4e79a7',edgecolor='black', alpha=0.8, capsize=5, label="Weighted Fusion")
avg = axs[1, 0].bar(x,y2, color="#9ea74e",edgecolor='black', alpha=0.8, capsize=5, label="Average Sum")
axs[1, 0].bar_label(bars, labels=bar_labels, padding=3, fontsize=18)
axs[1, 0].tick_params(axis='both', labelsize=18)
axs[1, 0].set_ylim(0, 1)
axs[1, 0].set_ylabel('SSIM', fontsize=18)
axs[1, 0].set_xlabel('(c)', fontsize=18)
axs[1, 0].legend()

# 第四个子图
x = ['Church', 'Bedroom', 'MS COCO']
y = np.array([0.8124, 0.6805, 0.5914])
y2 = np.array([0.6025, 0.4750, 0.4459])
delta= y-y2
bar_labels =[f'+{v:.4f}' for v in (delta)]

bars = axs[1, 1].bar(x,y, color='#4e79a7',edgecolor='black', alpha=0.8, capsize=5, label="Adaptive AE")
avg = axs[1, 1].bar(x,y2, color="#9ea74e",edgecolor='black', alpha=0.8, capsize=5, label="Fixed AE")
axs[1, 1].bar_label(bars, labels=bar_labels, padding=3, fontsize=18)
axs[1, 1].tick_params(axis='both', labelsize=18)
axs[1, 1].set_ylim(0.4, 1)
axs[1, 1].set_ylabel('Recall', fontsize=18)
axs[1, 1].set_xlabel('(d)', fontsize=18)
axs[1, 1].legend()

plt.tight_layout()#调整整体空白
plt.savefig("doc/fig.ablation2.svg", format = "svg",  transparent=True,dpi=600)
plt.savefig("doc/fig.ablation2.jpg", format = "jpg",  transparent=True,dpi=600)