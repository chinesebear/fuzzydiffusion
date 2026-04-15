import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/times.ttf')
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
import pandas as pd
import numpy as np


def f1(x):
    # 参数
    k = 10
    a = 430 #x
    b = 1 #y
    # 函数
    y = k / (x - a) + b
    return y

def f2(x):
    # 参数
    k = 10
    a = 370 #x
    b = 0.32 #y
    # 函数
    y = k / (x-a) + b
    return y

def f3(x):
    # 参数
    k = 10
    a = 560 #x
    b = 0 #y
    # 函数
    y = -k / (x - a) + b
    return y

def f4(x):
    # 参数
    k = 10
    a = 440 #x
    b = 0.34 #y
    # 函数
    y = -k / (x - a) + b
    return y

def f5(x):
    # 参数
    k = 8
    a = 670 #x
    b = 0.03 #y
    # 函数
    y = -k / (x - a) + b
    return y

def f6(x):
    # 参数
    k = 8
    a = 560 #x
    b = 0.34 #y
    # 函数
    y = -k / (x - a) + b
    return y


# 避开 x=0
x1 = np.linspace(1, 500, 500)
x2 = np.linspace(500, 1000, 500)
x = np.linspace(1, 1000, 1000)

# 函数
x1 = np.linspace(1, 400, 400)
x2 = np.linspace(400, 1000, 600)
y1 = np.concatenate((f1(x1), f2(x2)), axis=0)
x1 = np.linspace(1, 500, 500)
x2 = np.linspace(500, 1000, 500)
y2 = np.concatenate((f3(x1), f4(x2)), axis=0)
x1 = np.linspace(1, 600, 600)
x2 = np.linspace(600, 1000, 400)
y3 = np.concatenate((f5(x1), f6(x2)), axis=0)

# noise = np.random.normal(-0.02, 0.02, size=300)
# smooth_noise = np.convolve(noise, np.ones(10)/10, mode='same')
# y1[400:700] += smooth_noise
# y2[400:700] += smooth_noise
# y3[400:700] += smooth_noise

df = pd.DataFrame({'x': x, 'y1': y1, 'y2': y2, 'y3': y3})
df.to_csv("doc/csv/noalig_line.csv")
# 总样本数
N =len(df)
k = 50
# 均匀间隔索引
indices = np.linspace(0, N - 1, k, dtype=int)
# 采样
sampled_df = df.iloc[indices]
# 保存
sampled_df.to_csv('doc/csv/noalig_line_sampled_50.csv')

plt.figure()

# 双曲线
# plt.plot(x, y1, label='Rule Chain 1', color='skyblue')
# plt.plot(x, y2, label='Rule Chain 2', color='darkorange')
# plt.plot(x, y3, label='Rule Chain 3', color='slategrey')
plt.scatter(sampled_df['x'], sampled_df['y1'],color='skyblue')
plt.scatter(sampled_df['x'], sampled_df['y2'],color='darkorange')
plt.scatter(sampled_df['x'], sampled_df['y3'],color='slategrey')

# 渐近线（坐标轴）
plt.axhline(0.33, linestyle='--', color='red')  # y=0.33
# plt.axvline(400, linestyle='--')  # x=400
# plt.axvline(500, linestyle='--')  # x=500
# plt.axvline(600, linestyle='--')  # x=600

plt.xlim(0, 1000)
plt.ylim(0,1)



plt.tight_layout()#调整整体空白
plt.savefig("doc/fig.line2.shake.svg", format = "svg",  transparent=True,dpi=600)
plt.savefig("doc/fig.line2.shake.jpg", format = "jpg",  transparent=True,dpi=600)