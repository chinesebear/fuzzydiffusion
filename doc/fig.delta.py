import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/times.ttf')
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
import pandas as pd
import numpy as np


# 生成x
offset = 0
x = np.linspace(0+offset, 400+offset, 400)

# 正弦波
y = 0.2*np.sin(x/100)+0.2*np.sin(x/50)

# 画图
plt.plot(x, y)
plt.title("Sine Wave")
plt.xlabel("x")
plt.ylabel("sin(x)")


plt.tight_layout()#调整整体空白
plt.savefig("doc/fig.delta.svg", format = "svg",  transparent=True,dpi=600)
plt.savefig("doc/fig.delta.jpg", format = "jpg",  transparent=True,dpi=600)