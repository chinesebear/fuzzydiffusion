import matplotlib.font_manager as fm
fm.fontManager.addfont('doc/times/times.ttf')
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
import pandas as pd

df = pd.read_csv("doc/stable.csv")
epoch = df['Epoch']
dfs = df['FDM'] #DFS
md = df['MD']
sdg = df['SDG']
ldm = df['LDM']

dfs_t=epoch*4.1
md_t=epoch*4.7
sdg_t=epoch*1.5
ldm_t=epoch*1.1

plt.rcParams.update({'font.size': 18})
plt.subplots(1,2,figsize=(12,6))

plt.subplot(1,2,1)
plt.plot(epoch, dfs, label="DFS", color='red')
plt.plot(epoch, md, label="MD", color='tan')
plt.plot(epoch, sdg, label="SDG", color='skyblue')
plt.plot(epoch, ldm, label="LDM", color='slategrey')
plt.plot([0, 4], [dfs[4], dfs[4]], color='r', linestyle='dotted')
plt.plot([4, 4], [0, dfs[4]], color='blue', linestyle='dashed')
plt.scatter([4],[dfs[4]], color='r',zorder=3)
plt.text(2, dfs[4]+1, 'Ep:4', color='blue', fontsize=18)
plt.plot([5, 5], [0, 15], color='blue', linestyle='dashed')
plt.text(5, 15, 'Ep:5', color='blue', fontsize=18)
plt.scatter([5],[sdg[5]], color='skyblue',zorder=3)
plt.plot([0, 9], [ldm[9], ldm[9]], color='slategrey', linestyle='dotted')
plt.plot([9, 9], [0, 15], color='blue', linestyle='dashed')
plt.scatter([9],[ldm[9]], color='slategrey',zorder=3)
plt.text(9, 15, 'Ep:9', color='blue', fontsize=18)
plt.plot([13, 13], [0, 15], color='blue', linestyle='dashed')
plt.scatter([13],[sdg[13]], color='skyblue',zorder=3)
plt.text(13, 15, 'Ep:13', color='blue', fontsize=18)
plt.plot([20, 20], [0, 15], color='blue', linestyle='dashed')
plt.scatter([20],[sdg[20]], color='skyblue',zorder=3)
plt.scatter([20],[md[20]], color='tan',zorder=3)
plt.scatter([20],[ldm[20]], color='slategrey',zorder=3)
plt.text(20, 15, 'Ep:20', color='blue', fontsize=18)
plt.title("Training")
plt.xlabel("Epoch (Ep)")
plt.ylabel("FID")
plt.legend()
# plt.grid()
plt.text(12.5, -15, "(a)",fontsize=18, color='black')

plt.subplot(1,2,2)
plt.plot([4, 4], [0, 40], color='blue', linestyle='dashed')
plt.text(4, 40, f'Ep:4', color='blue',fontsize=18)
plt.plot([20,20], [0, 110], color='blue', linestyle='dashed')
plt.text(20, 110, f'Ep:20',color='blue', fontsize=18)
plt.plot(epoch, dfs_t, label="DFS", color='red')
plt.scatter([4],[dfs_t[4]], color='r',zorder=3)
plt.text(4, dfs_t[4]-6, f'{dfs_t[4]}h', fontsize=18)
plt.plot(epoch, md_t, label="MD", color='tan')
plt.scatter([20],[md_t[20]], color='tan',zorder=3)
plt.text(20, md_t[20], f'{md_t[20]}h', fontsize=18)
plt.plot(epoch, sdg_t, label="SDG", color='skyblue')
plt.scatter([20],[sdg_t[20]], color='skyblue',zorder=3)
plt.text(20, sdg_t[20]+4, f'{sdg_t[20]}h', fontsize=18)
plt.plot(epoch, ldm_t, label="LDM", color='slategrey')
plt.scatter([20],[ldm_t[20]], color='slategrey',zorder=3)
plt.text(20, ldm_t[20]-8, f'{ldm_t[20]}h', fontsize=18)
plt.title("Training")
plt.xlabel("Epoch (Ep)")
plt.ylabel("Time/h")
plt.legend()
# plt.grid()
plt.text(12.5, -40, "(b)",fontsize=18, color='black')

plt.tight_layout()
plt.savefig("doc/fig.convergence.svg", format = "svg", transparent=True)
plt.savefig("doc/fig.convergence.jpg", format = "jpg", transparent=True, dpi=300)
plt.show()