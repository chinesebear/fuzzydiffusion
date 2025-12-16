import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/times.ttf')
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
import pandas as pd
import numpy as np

# Label=["landspace","animal","human"]
# Color=["lightsteelblue","cornflowerblue","royalblue"]

df=pd.read_csv("doc/membershipcurve.csv")
landspace=np.flipud(df['landspace'].values[:50])
animal=np.flipud(df['animal'].values[:50])
human=np.flipud(df['human'].values[:50])
df2=pd.read_csv("doc/membershipcurve2.csv")
landspace2=np.flipud(df2['landspace'].values[:50])
animal2=np.flipud(df2['animal'].values[:50])
human2=np.flipud(df2['human'].values[:50])
df3=pd.read_csv("doc/membershipcurve3.csv")
landspace3=np.flipud(df3['landspace'].values[:50])
animal3=np.flipud(df3['animal'].values[:50])
human3=np.flipud(df3['human'].values[:50])
xlen=len(landspace)
x = np.flipud(np.array(range(xlen)))

fig, axe = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
plt.rcParams.update({'font.size': 14})

plt.subplot(1, 2, 1)
plt.scatter(x,landspace2,c="skyblue",marker='o',s=20, label='Rule Chain 1')
plt.scatter(x,animal2,c="darkorange",marker='o',s=20, label='Rule Chain 2')
plt.scatter(x,human2,c="slategrey",marker='o',s=20, label='Rule Chain 3')
plt.plot(x, 0.33*np.ones(xlen),c="red",linestyle='--',linewidth=1.0)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylim(0,1)
plt.xlabel("Diffusion Step",fontsize=18)
plt.ylabel("Membership Degree",fontsize=18)
# plt.grid()
plt.legend()
plt.text(25, -0.25, "(a)",fontsize=18, color='black')
plt.gca().invert_xaxis()

# plt.subplot(2, 2, 3)
# plt.bar(range(xlen),landspace2, color="skyblue", label='landspace', alpha=0.6, width=0.6)
# plt.bar(range(xlen), animal2, color="darkorange", label='animal', alpha=0.6, width=0.6, bottom=landspace2)
# plt.bar(range(xlen), human2, color="slategrey", label='human', alpha=0.6, width=0.6, bottom=landspace2+animal2)
# plt.xlabel("Diffusion Steps")
# plt.ylabel("Membership Distrubition")
# plt.ylim(0,1)
# plt.grid()
# # plt.legend(ncols=3, loc='upper center',fontsize=12)

plt.subplot(1, 2, 2)
plt.scatter(x,landspace3,c="skyblue",marker='o',s=20, label='Rule Chain 1')
# plt.plot(x,landspace3,c="skyblue")
plt.scatter(x,animal3,c="darkorange",marker='o',s=20, label='Rule Chain 2')
# plt.plot(range(xlen),animal3,c="darkorange")
plt.scatter(x,human3,c="slategrey",marker='o',s=20, label='Rule Chain 3')
# plt.plot(range(xlen),human3,c="slategrey")
plt.plot(x, 0.28*np.ones(xlen),c="red",linestyle='--',linewidth=1.0)
plt.xlabel("Diffusion Step",fontsize=18)
plt.ylabel("Membership Degree",fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylim(0,1)
# plt.grid()
plt.legend()
plt.text(25, -0.25, "(b)",fontsize=18, color='black')
plt.gca().invert_xaxis()

# plt.subplot(2, 2, 4)
# plt.bar(range(xlen),landspace3, color="skyblue", label='landspace', alpha=0.6, width=0.6)
# plt.bar(range(xlen), animal3, color="darkorange", label='animal', alpha=0.6, width=0.6, bottom=landspace3)
# plt.bar(range(xlen), human3, color="slategrey", label='human', alpha=0.6, width=0.6, bottom=landspace3+animal3)
# plt.xlabel("Diffusion Steps")
# plt.ylim(0,1)
# # plt.ylabel("Membership Distrubition")
# plt.grid()
# # plt.legend(ncols=3, loc='upper center',fontsize=12)


fig.tight_layout()#调整整体空白
plt.savefig("doc/fig.membership.svg", format = "svg", dpi=600)
plt.savefig("doc/fig.membership.jpg", format = "jpg", dpi=600)