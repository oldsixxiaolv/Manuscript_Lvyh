# -- coding:utf-8 --
from shuju_process import read_shuju
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from color_change_newest import color_characterized


def index(x):
    index1 = list(np.where(x))


"""提取数据部分"""
plt.rcParams["font.size"] = 50
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('axes', linewidth=3)
# 参数projection让ax从普通的axes变成了GeoAxes
data = read_shuju()
maxht20 = data[0]
maxht40 = data[1]
print(sorted(list(maxht20)))
"""绘图部分"""
# 这里的ccrs.PlateCarree是一个转换器
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(direction='in', which="major", length=15, width=3)
ax.minorticks_on()
ax.tick_params(direction='in', which="minor", length=12, width=2)
# ax.text(0, 0.01, "Maxht20/km", transform=ax.transAxes, font={"family": "Times New Roman",
#                                                              "size": 30, "weight": "bold"})
# 这里的np.vstack函数是让maxht40和maxht20变成一个二维数组
# x_and_y = np.vstack([maxht40, maxht20])
# # 这里的gaussian_kde是高斯核估计，
# kde = gaussian_kde(x_and_y)
# z = kde(x_and_y)
# idx = z.argsort()
# x, y, z = maxht40[idx], maxht20[idx], z[idx]
x = maxht40
y = maxht20
z1 = np.polyfit(x, y, 1)
p1 = np.poly1d(z1)
slope, intercept = z1[0], z1[1]
correlation = np.corrcoef(x, y)[0, 1]
equation_text = f'equation: y = {slope:.2f} x + {intercept:.2f}'
correlation_text = f'r = {correlation:.2f}'
# ax.text(0.03, 0.2, equation_text, transform=ax.transAxes, fontsize=40)
ax.text(0.775, 0.23, correlation_text, transform=ax.transAxes, fontsize=50)
ax.text(0.05, 0.88, "(b)", transform=ax.transAxes, fontsize=70)
w = 0
max40, max20, _color_, allshuju, den10,\
    den20, den30, den40, den50, den60, den70, den80, den90 = color_characterized(maxht40, maxht20, w)
_num_ = []
for i in sorted(list(set(maxht40))):
    _num_mid = []
    for j in sorted(list(set(maxht20))):
        index1 = list(np.where(np.array(maxht40) == i))
        index2 = list(np.where(np.array(maxht20) == j))
        index = list(set(index1[0]) & set(index2[0]))
        density = len(maxht40[index])
        _num_mid.append(density)
    _num_.append(_num_mid)
color = ["grey", "deepskyblue", "green",
         "darkorange", "r"]
axx = ax.contour(sorted(list(set(maxht40))), sorted(list(set(maxht20))),
                 np.array(_num_).T, colors=color,
                 levels=[25, 50, 100, 200, 400],
                 extend="both", linewidths=5)
plt.clabel(axx, inline=True, colors="black", fontsize=37)
ax.scatter(max40, max20, s=200, color=_color_, marker=".")
# ax.text(0.03, 0.08, f"Total = {allshuju}", transform=ax.transAxes, fontsize=50)
# 把斜率加到右下角
ax.text(0.775, 0.155, f"k = {round(slope, 2)}", transform=ax.transAxes, fontsize=50)
ax.text(0.775, 0.08, f"b = {round(intercept, 2)}", transform=ax.transAxes, fontsize=50)
x_vals = np.array([2.5, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
y_vals = intercept + slope * x_vals
ax.plot(x_vals, y_vals, color="black", linewidth=6)
ax.vlines(sorted(max40)[round(len(max40) * 0.9)], 0, 21, linestyles="dashed",
          color="black", label=f"Maxht40={sorted(max40)[round(len(max40) * 0.9)]} ,  90%",
          linewidth=4)
ax.vlines(sorted(max40)[round(len(max40) * 0.8)], 0, 21, linestyles="dashed",
          color="black", label=f"Maxht40={sorted(max40)[round(len(max40) * 0.8)]} ,  80%",
          linewidth=4)
ax.vlines(sorted(max40)[round(len(max40) * 0.5)], 0, 21, linestyles="dashed",
          color="black", label=f"Maxht40={sorted(max40)[round(len(max40) * 0.5)]} ,  50%",
          linewidth=4)
ax.hlines(sorted(max20)[round(len(max20) * 0.9)], 0, 21, linestyles="dashed",
          color="black", label=f"Maxht20={sorted(max40)[round(len(max40) * 0.9)]} ,  90%",
          linewidth=4)
ax.hlines(sorted(max20)[round(len(max20) * 0.8)], 0, 21, linestyles="dashed",
          color="black", label=f"Maxht20={sorted(max40)[round(len(max40) * 0.8)]} ,  80%",
          linewidth=4)
ax.hlines(sorted(max20)[round(len(max20) * 0.5)], 0, 21, linestyles="dashed",
          color="black", label=f"Maxht20={sorted(max40)[round(len(max40) * 0.5)]} ,  50%",
          linewidth=4)
ax.set_xticks([2, 4, 6, 8, 10, 12, 14, 16])
ax.set_yticks([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
# ax.scatter(maxht40, maxht20, cmap="viridis", c=z)
ax.set_xlim(2.01, 16)
ax.set_ylim(2.01, 21)
ax.set_xlabel("Maxht40 (km)", fontsize=60)
ax.set_ylabel("Maxht20 (km)", fontsize=60)
ax.set_title("Americas", fontsize=70)
plt.savefig(r"C:\Users\lvyih\Desktop\nanmeizhou_sandiantu_maxht20_40_setlim.jpeg", bbox_inches="tight", dpi=500)
# color=["grey", "cornflowerblue", "forestgreen", "yellow", "peru", "firebrick"]
