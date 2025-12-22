# -- coding:utf-8 --
from pyhdf.SD import SD, SDC
import numpy as np
import matplotlib.pyplot as plt
from color_characterized import color_characterized
import matplotlib.colors as mcolors


def duqu_excel():
    import pandas as pd
    file_path = r"C:\Users\lvyih\Desktop\duqu_spss.xlsx"
    df = pd.read_excel(file_path, usecols=[1], names=None)
    dali = df.values.tolist()
    result = []
    for i in dali:
        result.append(i[0])
    return result


def npixels_size(npixels, boost):
    boost_former = 4.3 * 4.3
    boost_latter = 5 * 5
    npixel = list(npixels)
    pixel = []
    boost = list(boost)
    t = zip(boost, npixel)
    # 记住以下我们以后要是碰到需要利用两个列表同时遍历的情况，一定要使用zip
    for i, j in t:
        if i == 1:
            mid = j * boost_latter
            pixel.append(mid)
        else:
            mid = j * boost_former
            pixel.append(mid)
    return np.array(pixel)


"""读取数据"""


def read_shuju(former, latter):
    data = SD('C:\\Users\\lvyih\\Desktop\\TRMM_tropical_convection_dataset.hdf', SDC.READ)
    longitude = list(data.select("longitude")[:])
    latitude = list(data.select("latitude")[:])
    landocean = list(data.select("landocean")[:])
    longitude = np.array(list(map(int, longitude)))
    latitude = np.array(list(map(int, latitude)))
    landocean = np.array(list(map(int, landocean)))
    flashcount = data.select('flashcount')[:]
    flashcount = np.array(list(map(int, flashcount)))
    maxht40 = data.select("maxht40")[:]
    maxht40 = np.array(list(map(int, maxht40)))
    rainconv = data.select('rainconv')[:]
    volrain = data.select('volrain')[:]
    r = np.divide(rainconv, volrain)
    # 筛选数据
    index1 = np.where(longitude > -20)
    index2 = np.where(latitude > -20)
    index3 = np.where(landocean == 1)
    index4 = np.where(longitude < 50)
    index5 = np.where(latitude < 20)
    # 把没有闪电数据的但是有maxht40的和没有maxht40但有闪电数据的去除
    index6 = list(np.where(flashcount != 0))
    index7 = list(np.where(maxht40 != 0))
    # index8 = list(np.where(r < 0.98))
    # index9 = list(np.where(r >= 0.8))
    # 两个数组取交集
    index = list(set(index1[0]) & set(index2[0]) & set(index3[0]) & set(index4[0]) & set(index5[0]) &
                 set(index7[0]))
    #  & set(index8[0]) & set(index9[0])
    index = sorted(index)
    r = r[index]
    # 把hdf文件中的闪电频数数据提取出来
    flashcount = data.select('flashcount')[:]
    flashcount = flashcount[index]
    # 把hdf文件中总的降水数据提取出来
    volrain = data.select('volrain')[:]
    volrain = volrain[index]
    # 把hdf文件中的层云降水数据提取出来
    rainstrat = data.select('rainstrat')[:]
    rainstrat = rainstrat[index]
    # 把hdf文件中的对流降水数据分离出
    rainconv = data.select('rainconv')[:]
    rainconv = rainconv[index]
    # 把hdf文件中的观测时间数据提取出来
    viewtime = data.select('viewtime')[:]
    viewtime = viewtime[index]
    # 把hdf文件是否升轨数据提取出来
    boost = data.select("boost")[:]
    boost = boost[index]
    # 把hdf文件的最大顶高数据提取出来（这个数据老师说第一个maxht不用操作）
    maxht = data.select("maxht")[:]
    maxht = maxht[index]
    maxht20 = data.select("maxht20")[:]
    maxht20 = maxht20[index]
    maxht30 = data .select("maxht30")[:]
    maxht30 = maxht30[index]
    maxht40 = data.select("maxht40")[:]
    maxht40 = maxht40[index]
    # 这个是云的顶层的亮温，可以通过知晓这个温度的高低来判断云顶的高度，温度越低意味着云越高
    minir = data.select("minir")[:]
    minir = minir[index]
    # 这个先不用
    npixels_40 = data.select("npixels_40")[:]
    npixels_40 = npixels_40[index]
    npixels_40 = npixels_size(npixels_40, boost)
    npixels_20 = data.select("npixels_20")[:]
    npixels_20 = npixels_20[index]
    npixels_20 = npixels_size(npixels_20, boost)
    npixels_30 = data.select("npixels_30")[:]
    npixels_30 = npixels_30[index]
    npixels_30 = npixels_size(npixels_30, boost)
    maxnsrain = data.select("maxnsrain")[:]
    maxnsrain = maxnsrain[index]
    flashcount = flashcount.astype(float)
    flashrate = np.divide(flashcount, viewtime)
    flashrate = np.multiply(flashrate, 60)
    inde = list(np.where(r > former))
    ind = list(np.where(r <= latter))
    inded = list(set(ind[0]) & set(inde[0]))
    inded = sorted(inded)
    flashcount = flashcount[inded]
    flashrate = flashrate[inded]
    maxht40 = maxht40[inded]
    maxht20 = maxht20[inded]
    maxht30 = maxht30[inded]
    npixels_20 = npixels_20[inded]
    npixels_30 = npixels_30[inded]
    npixels_40 = npixels_40[inded]
    return flashrate, maxht20, maxht30, maxht40, npixels_20, npixels_30, npixels_40, flashcount


def four_seven_eight(x):
    list_mid = list(x)
    list_mid.sort()
    four = list_mid[round(len(list_mid) * 0.4)]
    eight = list_mid[round(len(list_mid) * 0.8)]
    return four, eight


fig = plt.figure(1, figsize=(5, 5))
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
data_test = read_shuju(0.64, 1)
flashrate = data_test[0]
maxht20 = data_test[1]
maxht30 = data_test[2]
maxht40 = data_test[3]
npixels_20 = data_test[4]
npixels_30 = data_test[5]
npixels_40 = data_test[6]
flashcount = data_test[7]
maxht30_mid = four_seven_eight(maxht30)
maxht40_mid = four_seven_eight(maxht40)
maxht40_40 = maxht40_mid[0]
maxht40_80 = maxht40_mid[1]
maxht30_40 = maxht30_mid[0]
maxht30_80 = maxht30_mid[1]
ax = fig.add_subplot(1, 1, 1)
w = 0
max40, max30, _color_, allshuju = color_characterized(flashcount, maxht40, maxht30, w)
ax.scatter(max40, max30, s=2, color=_color_, marker=".")
ax.hlines(maxht30_40, 0, 20, linestyles="dashed", color="black", label="40%")
ax.hlines(maxht30_80, 0, 20, linestyles="dashed", color="red", label="80%")
# 创建绘制等值线图
x = np.arange(0, 20, 0.25)
y = np.arange(0, 20, 0.25)
all_scale = []
for i in y:
    two_scale = []
    for j in x:
        indexx = list(np.where(maxht30 == i))
        indexxx = list(np.where(maxht40 == j))
        indexq = list(set(indexx[0]) & set(indexxx[0]))
        two_scale.append(len(indexq))
    all_scale.append(two_scale)
# levels = [0, 400, 800, 1200, 1600, 2000]
# cmap, norm = mcolors.from_levels_and_colors(levels, ["darkgreen", "blueviolet", "b", "cyan",
#                                                      "yellow", "orange", "r"], "both")
# axx = ax.contour(x, y, all_scale, cmap=cmap, norm=norm, extend="both", linewidths=1.1)
axx = ax.contour(x, y, all_scale, colors="b", extend="both", linewidths=0.5)
# plt.clabel(axx, inline=True, colors="black", fmt="%1.2f", fontsize=5)
plt.clabel(axx, levels=[400, 800, 1200], inline=True, colors="black", fontsize=10)
# fig.colorbar(axx, ax=ax, extend="both")
ax.set_xlabel("Maxht40/km", fontsize=14)
ax.set_ylabel("Maxht30/km", fontsize=14)
ax.set_yticks([0, 5, 10, 15, 20])
ax.set_xlim(0, 20)
ax.set_ylim(0, 20)
# ax.set_title("maxht20 and maxht40", fontsize=14)
# ax.vlines(maxht40_40, 0, 20, linestyles="dashed", color="black", label="40%")
# ax.vlines(maxht40_80, 0, 20, linestyles="dashed", color="red", label="80%")
ax.text(0.7, 0.05, f"Total={allshuju}", transform=ax.transAxes, fontsize=13)
# 将上下左右的主刻度线都打开
ax.tick_params(direction='in', which="both", top=True, right=True)
ax.legend(loc="upper left", prop={"family": "Times New Roman", "size": 13})
ax.minorticks_on()
# 这是打开次刻度的代码
plt.savefig(r"C:\\Users\\lvyih\\Desktop\\all_stage____.jpeg", dpi=400)
