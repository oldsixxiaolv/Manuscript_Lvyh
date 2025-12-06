# -- coding:utf-8 --
from shuju_process import read_shuju
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
"""提取数据部分"""
plt.rcParams["font.size"] = 30
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('axes', linewidth=2)
# 参数projection让ax从普通的axes变成了GeoAxes
data = read_shuju()
lat = data[0]
lon = data[1]
# df = pd.DataFrame({"latitude": lat, "longitude": lon})
# df.to_excel(r"C:\Users\lvyih\Desktop\matlab.xlsx", sheet_name="sheet1", index=False)
mid1 = []
mid2 = []
for i in lat:
    i = round(i)
    mid1.append(i)
for j in lon:
    j = round(j)
    mid2.append(j)
lat = np.array(mid1)
lon = np.array(mid2)
all_dot = []
for i in range(-22, 23):
    mid = []
    for j in range(-20, 51):
        k = 0
        index1 = list(np.where(lat == i))
        longitu = lon[index1][0]
        for q in longitu:
            if q == j:
                k += 1
        mid.append(k)
    all_dot.append(mid)
latitude = [i for i in range(-22, 23)]
longitude = [j for j in range(-20, 51)]
"""绘图部分"""
# 这里的ccrs.PlateCarree是一个转换器
fig = plt.figure(figsize=(10, 15))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax2 = ax.gridlines(draw_labels=True)
ax2.rotate_labels = False
ax2.top_labels = False
ax2.right_labels = False
# colormap = plt.get_cmap("jet")
levels = [0, 1, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300]
cmap, norm = mcolors.from_levels_and_colors(levels, ["white", "royalblue",
                                                     "cyan", "lime", "limegreen", "greenyellow",
                                                     "goldenrod",  "red", "firebrick",
                                                     "slateblue", "purple", "indigo"], extend="max")

# contourf是填充的图，而contour是线图, 这里的extent="both"可以表示上限和下限
ax1 = ax.contourf(longitude, latitude, all_dot, levels=levels, cmap=cmap, norm=norm,
                  linewidths=1, transfrom=ccrs.PlateCarree(), extend="max")
ax.add_feature(cfeature.COASTLINE.with_scale("110m"))
ax.text(-19, -8.5, f"Total={len(lat)}",
        font={"family": "Times New Roman", "size": 40, "weight": "bold"})
ax.text(-9, -15, f"{round(round(len(lat) / 135089, 2) * 100)}%",
        font={"family": "Times New Roman", "size": 40, "weight": "bold"})
ax.text(-19, 24, "(c)", font={"family": "Times New Roman", "size": 50, "weight": "bold"})
# ax.set_title("Tropical africa Multi-cell storm distribution", font={"family": "Times New Roman", "size": 35})
# 这个figbar我们用我们ppt中的
# fig.colorbar(ax1, ax=ax, pad=0.07,  extend="max")
plt.tight_layout()
plt.savefig(r"C:\Users\lvyih\Desktop\Tropical_africa_Multi_cell_distribution.jpeg", bbox_inches="tight", dpi=500)
