import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/times.ttf')
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
import pandas as pd
import numpy as np

Method=["LDM","SDG","MD","DFS"]
Color=["slategrey","skyblue","tan","red"]

Scale=[1537,1571,5210,3516] # M
Train_Time_Epoch=[8,8,43,19] #h
Gen_Time_100=[20,21,23,15] #s

bar_width = 0.6
font_size = 14

data=pd.read_csv('doc/train.csv').values

CPU=data[:4]
RAM=data[4:8].mean(axis=1)
GPU=data[8:12]
VRAM=data[12:].mean(axis=1)

data=pd.read_csv('doc/test.csv').values

CPU2=data[:4]
RAM2=data[4:8].mean(axis=1)
GPU2=data[8:12]
VRAM2=data[12:].mean(axis=1)

fig, axe = plt.subplots(nrows=3, ncols=4, figsize=(12, 8),dpi=600)
# plt.rcParams.update({'font.size': 18})

# 第一行
plt.subplot(3, 4, 1)
bar_labels =[f'{v:.0f}M' for v in (Scale)]
bars = plt.bar(Method, Scale, width=bar_width, color=Color, alpha=0.7, label=Method)
plt.bar_label(bars, labels=bar_labels, padding=3)
plt.ylabel('Parameter (M)')
plt.ylim(0, 8000)
plt.xlabel('(a) Model Scale',fontsize=font_size)
plt.legend(ncols=2, loc='upper left')
# plt.grid()

plt.subplot(3, 4, 2)
bar_labels =[f'{v:.1f}h' for v in (Train_Time_Epoch)]
bars = plt.bar(Method, Train_Time_Epoch,color=Color, width=bar_width, alpha=0.7, label=Method)
plt.bar_label(bars, labels=bar_labels, padding=3)
plt.ylim(0, 70)
plt.ylabel('Train Time @epoch (h)')
plt.xlabel('(b) Train Time',fontsize=font_size)
plt.legend(ncols=2, loc='upper left')
# plt.grid()

plt.subplot(3, 4, 3)
bar_labels =[f'{v:.0f}s' for v in (Gen_Time_100)]
bars = plt.bar(Method, np.array(Gen_Time_100),color=Color, width=bar_width, alpha=0.7, label=Method)
plt.bar_label(bars, labels=bar_labels, padding=3)
plt.ylim(10,40)
plt.ylabel('Generation Time @100 (s)')
plt.xlabel('(c) Generation Time',fontsize=font_size)
plt.legend(ncols=2, loc='upper left')

plt.subplot(3, 4, 4)
data_scale= Scale[3]/Scale[2]*100
data_train_time_epoch = Train_Time_Epoch[3] / Train_Time_Epoch[2] * 100
data_Gen_Time_100 = Gen_Time_100[3] / Gen_Time_100[2] * 100
data = [data_scale, data_train_time_epoch, data_Gen_Time_100]
# delta_scale= (Scale[3]-Scale[2])/Scale[2]*100
# delta_train_time_epoch = (Train_Time_Epoch[3] - Train_Time_Epoch[2]) / Train_Time_Epoch[2] * 100
# delta_Gen_Time_100 = (Gen_Time_100[3] - Gen_Time_100[2]) / Gen_Time_100[2] * 100
# delta_data = [delta_scale, delta_train_time_epoch, delta_Gen_Time_100]
bar_labels =[f'{v:.1f}%' for v in data]
delta_method = ['Scale', 'Train Time', 'Gen Time']
bars = plt.bar(delta_method, np.array(data), width=bar_width, alpha=0.7, label=delta_method,color=['#4C72B0','#6FA8DC','#A6C8E0'])
plt.bar_label(bars, labels=bar_labels, padding=3)
plt.ylim(40,100)
plt.ylabel('Ratio (%)')
plt.xlabel('(d) DFS vs MD',fontsize=font_size)
plt.legend(ncols=1, loc='upper left')







#第二行

plt.subplot(3, 4, 5)
bars = plt.bar(Method, np.array(RAM),color=Color, width=bar_width,alpha=0.7, label=Method)
plt.bar_label(bars, labels=[f'{v:.0f}MB' for v in RAM], padding=3)
plt.ylim(200,2000)
plt.ylabel('RAM (MB)')
plt.xlabel('(e) RAM (Train)',fontsize=font_size)
plt.legend(ncols=2, loc='upper left')

plt.subplot(3, 4, 6)
bar_labels =[f'{v:.0f}MB' for v in (RAM2)]
bars = plt.bar(Method, np.array(RAM2),color=Color,width=bar_width, alpha=0.7, label=Method)
plt.bar_label(bars, labels=bar_labels, padding=3)
plt.ylim(100,600)
plt.ylabel('RAM (MB)')
plt.xlabel('(f) RAM (Generation)',fontsize=font_size)
plt.legend(ncols=2, loc='upper left')

plt.subplot(3, 4, 7)
bar_labels =[f'{v:.1f}GB' for v in (VRAM)]
bars = plt.bar(Method, np.array(VRAM), color=Color, width=bar_width, alpha=0.7, label=Method)
plt.bar_label(bars, labels=bar_labels, padding=3)
plt.ylim(0,48)
plt.ylabel('VRAM (GB)')
plt.xlabel('(g) VRAM (Train)',fontsize=font_size)
plt.legend(ncols=2, loc='upper left')



plt.subplot(3, 4, 8)
bar_labels =[f'{v:.1f}GB' for v in (VRAM2)]
bars =plt.bar(Method, np.array(VRAM2),color=Color, width=bar_width, alpha=0.7, label=Method)
plt.bar_label(bars, labels=bar_labels, padding=3)
plt.ylim(2.5,20)
plt.ylabel('VRAM (GB)')
plt.xlabel('(h) VRAM (Generation)',fontsize=font_size)
plt.legend(ncols=2, loc='upper left')


# 第三行
plt.subplot(3, 4, 9)
plt.plot(CPU[0], label=Method[0], color=Color[0])
plt.plot(CPU[1], label=Method[1], color=Color[1])
plt.plot(CPU[2], label=Method[2], color=Color[2])
plt.plot(CPU[3], label=Method[3], color=Color[3])
plt.xlabel('Epoch')
plt.ylabel('CPU Occ (%)')
plt.ylim(0,40)
plt.xlabel('(i) CPU (Train)',fontsize=font_size)
plt.legend(ncols=2, loc='upper left')
# plt.grid()


plt.subplot(3, 4, 10)
plt.plot(CPU2[0], label=Method[0], color=Color[0])
plt.plot(CPU2[1], label=Method[1], color=Color[1])
plt.plot(CPU2[2], label=Method[2], color=Color[2])
plt.plot(CPU2[3], label=Method[3], color=Color[3])
plt.xlabel('Times')
plt.ylabel('CPU Occ (%)')
plt.ylim(0,5)
plt.xlabel('(j) CPU (Generation)',fontsize=font_size)
plt.legend(ncols=2, loc='upper left')

plt.subplot(3, 4, 11)
plt.plot(GPU[0], label=Method[0], color=Color[0])
plt.plot(GPU[1], label=Method[1], color=Color[1])
plt.plot(GPU[2], label=Method[2], color=Color[2])
plt.plot(GPU[3], label=Method[3], color=Color[3])
plt.xlabel('Epoch')
plt.ylabel('GPU Occ (%)')
plt.ylim(0,100)
plt.xlabel('(k) GPU (Train)',fontsize=font_size)
plt.legend(ncols=2, loc='lower left')



plt.subplot(3, 4, 12)
plt.plot(GPU2[0], label=Method[0], color=Color[0])
plt.plot(GPU2[1], label=Method[1], color=Color[1])
plt.plot(GPU2[2], label=Method[2], color=Color[2])
plt.plot(GPU2[3], label=Method[3], color=Color[3])
plt.xlabel('Times')
plt.ylabel('GPU Occ (%)')
plt.ylim(0,100)
plt.xlabel('(l) GPU (Generation)',fontsize=font_size)
plt.legend(ncols=2, loc='upper left')


plt.tight_layout()
plt.savefig("doc/fig.compute.svg", format = "svg", transparent=True, dpi=600)
plt.savefig("doc/fig.compute.jpg", format = "jpg", transparent=True, dpi=600)
plt.show()