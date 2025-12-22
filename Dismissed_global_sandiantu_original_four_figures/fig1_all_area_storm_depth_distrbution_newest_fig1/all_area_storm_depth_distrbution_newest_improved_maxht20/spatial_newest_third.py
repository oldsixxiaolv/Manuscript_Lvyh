# -- coding:utf-8 --
from shuju_process import read_shuju
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import matplotlib as mpl
import pandas as pd
import matplotlib.colors as mcolors
from cartopy.mpl.ticker import LongitudeFormatter
from cartopy.mpl.ticker import LatitudeFormatter


"""提取数据部分"""
plt.rcParams["font.size"] = 30
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('axes', linewidth=2)
# plt.rcParams["axes.spines.top"] = False
# plt.rcParams["axes.spines.right"] = False
# 参数projection让ax从普通的axes变成了GeoAxes
data = read_shuju()
lat = data[0]
lon = data[1]
maxht20 = data[2]
# print(type(maxht20))
minir = data[3]
maxht40 = data[4]
flashrate = data[5]
# df = pd.DataFrame({"latitude": lat, "longitude": lon})
# df.to_excel(r"C:\Users\lvyih\Desktop\matlab.xlsx", sheet_name="sheet1", index=False)
mid1 = []
mid2 = []
for i in lat:
    i = round(i, 2)
    mid1.append(i)
for j in lon:
    j = round(j, 2)
    mid2.append(j)
lat = np.array(mid1)
lon = np.array(mid2)
color = []
maxht20_mid = sorted(maxht20)
for q in maxht20:
    if q < maxht20_mid[round(len(maxht20_mid) * 0.5)]:
        color.append("grey")
    elif maxht20_mid[round(len(maxht20_mid) * 0.5)] <= q < maxht20_mid[round(len(maxht20_mid) * 0.75)]:
        color.append("cornflowerblue")
    elif maxht20_mid[round(len(maxht20_mid) * 0.75)] <= q < maxht20_mid[round(len(maxht20_mid) * 0.9)]:
        color.append("forestgreen")
    elif maxht20_mid[round(len(maxht20_mid) * 0.9)] <= q < maxht20_mid[round(len(maxht20_mid) * 0.99)]:
        color.append("gold")
    elif maxht20_mid[round(len(maxht20_mid) * 0.99)] <= q < maxht20_mid[round(len(maxht20_mid) * 0.995)]:
        color.append("chocolate")
    else:
        color.append("firebrick")
print(maxht20_mid[round(len(maxht20_mid) * 0.5)])
print(maxht20_mid[round(len(maxht20_mid) * 0.75)])
print(maxht20_mid[round(len(maxht20_mid) * 0.9)])
print(maxht20_mid[round(len(maxht20_mid) * 0.99)])
print(maxht20_mid[round(len(maxht20_mid) * 0.995)])
# print(maxht20_mid)
"""绘图部分"""
# 这里的ccrs.PlateCarree是一个转换器
fig = plt.figure(figsize=(25, 10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax2 = ax.gridlines(draw_labels=True, ls="--",
                   xlocs=[-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180])
# 设置不显示上面和右边的刻度标签
ax2.top_labels = False
ax2.right_labels = False
ax2.rotate_labels = False
ax.add_feature(cfeature.COASTLINE.with_scale("110m"))
ax.text(0, 1.05, "(a)  Maxht20 (km)", transform=ax.transAxes, color="black", font={"family": "Times New Roman",
                                                                                   "size": 45, "weight": "bold"})
levels = [10.75, 12.25, 14.25, 15.50, 16.00]
color_map = ["grey", "cornflowerblue", "forestgreen", "gold", "chocolate", "firebrick"]
cmap, norm = mcolors.from_levels_and_colors(levels, color_map, extend="both")
color = np.array(color)
for w in color_map:
    index_mid = list(np.where(np.array(color) == w))
    index_mid = list(set(index_mid[0]))
    index_mid = sorted(index_mid)
    lat_mid = lat[index_mid]
    lon_mid = lon[index_mid]
    color_mid = color[index_mid]
    ax.scatter(lon_mid, lat_mid, c=color_mid, transform=ccrs.PlateCarree(), s=2)
np.set_printoptions(threshold=np.inf)
# ax.text(-25, 33, "(a)", font={"family": "Times New Roman", "size": 30, "weight": "bold"})
# ax.set_title("Tropical all storm distribution", font={"family": "Times New Roman", "size": 30})
# fig.colorbar(ax1, ax=ax, orientation="horizontal", cmap=cmap, norm=norm,
#              fraction=0.02, aspect=175, location="top", pad=2)
plt.tight_layout()
plt.savefig(r"C:\Users\lvyih\Desktop\Tropical_africa_all_storm_distribution.jpeg", bbox_inches="tight", dpi=100)
# color=["grey", "cornflowerblue", "forestgreen", "yellow", "peru", "firebrick"]
