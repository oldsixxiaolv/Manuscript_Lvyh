# -- coding:utf-8 --
from pyhdf.SD import SD, SDC
import numpy as np
import matplotlib.pyplot as plt
import math


def duqu_excel():
    import pandas as pd
    # temp not write (not nessary)
    file_path = r'' 
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


def read_shuju():
    data = SD('/root/git/Project_develop/TRMM_tropical_convection_dataset.hdf', SDC.READ)
    longitude = data.select("longitude")[:]
    latitude = data.select("latitude")[:]
    landocean = list(data.select("landocean")[:])
    landocean = np.array(list(map(int, landocean)))
    flashcount = data.select('flashcount')[:]
    maxht20 = data.select("maxht20")[:]
    volrain = data.select('volrain')[:]
    rainconv = data.select('rainconv')[:]
    npixels_20 = data.select("npixels_20")[:]
    npixels_40 = data.select("npixels_40")[:]
    viewtime = data.select("viewtime")[:]
    maxht40 = data.select("maxht40")[:]
    # boost = data.select("boost")[:]
    # npixels_40_area = npixels_size(npixels_40, boost)
    r = np.divide(rainconv, volrain)
    # 筛选数据
    index1 = np.where(longitude > -20)
    index2 = list(np.where(latitude > -20))
    index3 = list(np.where(landocean == 1))
    index4 = list(np.where(longitude < 50))
    index5 = list(np.where(latitude < 20))
    # 把没有闪电数据的但是有maxht40的和没有maxht40但有闪电数据的去除
    # index6 = list(np.where(flashcount != 0))
    index6 = list(np.where(npixels_40 != 0))
    index7 = list(np.where(maxht40 != 0))
    index8 = list(np.where(flashcount != 0))
    # index8 = list(np.where(npixels_20 > 0))
    # index9 = list(np.where(r > 0))
    # index10 = list(np.where(r < 1))
    index11 = list(np.where(maxht20 != 0))
    index12 = list(np.where(npixels_20 != 0))
    index13 = list(np.where(~np.isnan(viewtime)))
    # npixels_40 = data.select("npixels_40")[:]
    # 两个数组取交集
    index = list(set(index1[0]) & set(index2[0]) & set(index3[0]) & set(index4[0]) &
                 set(index5[0]) & set(index6[0]) & set(index7[0]) & set(index8[0]) &
                 set(index11[0]) & set(index12[0]) & set(index13[0]))
    index = sorted(index)
    # indexx = duqu_excel()
    # 把hdf文件中的闪电频数数据提取出来
    flashcount = data.select('flashcount')[:]
    """这里我们可以看到对于一个可迭代的变量索引可以使用一个列表然后就可以直接赋值。"""
    flashcount = flashcount[index]
    # flashcount = flashcount[indexx]
    # 把hdf文件中总的降水数据提取出来
    volrain = data.select('volrain')[:]
    volrain = volrain[index]
    # volrain = volrain[indexx]
    # 把hdf文件中的对流降水数据分离出来
    rainconv = data.select('rainconv')[:]
    rainconv = rainconv[index]
    # rainconv = rainconv[indexx]
    # 把hdf文件中的观测时间数据提取出来
    viewtime = data.select('viewtime')[:]
    viewtime = viewtime[index]
    # viewtime = viewtime[indexx]
    # 把hdf文件是否升轨数据提取出来
    boost = data.select("boost")[:]
    boost = boost[index]
    # boost = boost[indexx]
    minir = data.select("minir")[:]
    minir = minir[index]
    # minir = minir[indexx]
    maxht20 = data.select("maxht20")[:]
    maxht20 = maxht20[index]
    # maxht20 = maxht20[indexx]
    maxht30 = data.select("maxht30")[:]
    maxht30 = maxht30[index]
    # maxht30 = maxht30[indexx]
    maxht40 = data.select("maxht40")[:]
    maxht40 = maxht40[index]
    # maxht40 = maxht40[indexx]
    npixels_40 = data.select("npixels_40")[:]
    npixels_40 = npixels_40[index]
    # npixels_40 = npixels_40[indexx]
    npixels_40 = npixels_size(npixels_40, boost)
    npixels_40_K = np.multiply(npixels_40, 4)
    npixels_40_K = np.divide(npixels_40_K, math.pi)
    npixels_40_R = np.sqrt(npixels_40_K)
    npixels_30 = data.select("npixels_30")[:]
    npixels_30 = npixels_30[index]
    # npixels_30 = npixels_30[indexx]
    npixels_30 = npixels_size(npixels_30, boost)
    npixels_30_K = np.multiply(npixels_30, 4)
    npixels_30_K = np.divide(npixels_30_K, math.pi)
    npixels_30_R = np.sqrt(npixels_30_K)
    npixels_20 = data.select("npixels_20")[:]
    npixels_20 = npixels_20[index]
    # npixels_20 = npixels_20[indexx]
    npixels_20 = npixels_size(npixels_20, boost)
    npixels_20_K = np.multiply(npixels_20, 4)
    npixels_20_K = np.divide(npixels_20_K, math.pi)
    npixels_20_R = np.sqrt(npixels_20)
    # n20和n40
    n40 = data.select("n40dbz")[:]
    n40 = n40[index]
    n40_mid = []
    for i, j in zip(n40, boost):
        if j == 1:
            mid = sum(i * 5 * 5 * 1.25)
        else:
            mid = sum(i * 4.3 * 4.3 * 1.25)
        n40_mid.append(mid)
    n40_volume = np.array(n40_mid)
    n20 = data.select("n20dbz")[:]
    n20 = n20[index]
    n20_mid = []
    for i, j in zip(n20, boost):
        if j == 1:
            mid = sum(i * 5 * 5 * 1.25)
        else:
            mid = sum(i * 4.3 * 4.3 * 1.25)
        n20_mid.append(mid)
    n20_volume = np.array(n20_mid)
    # 闪电频数=闪电数/viewtime*60，我们使用每分钟的闪电频数
    flashfrequence = np.divide(flashcount * 60, viewtime)
    maxdbz = data.select("maxdbz")[:]
    maxdbz = maxdbz[index]
    # maxdbz = maxdbz[indexx]
    maxht = data.select("maxht")[:]
    maxht = maxht[index]
    # maxht = maxht[indexx]
    flash_20 = np.divide(flashfrequence * 100, npixels_20)
    flash_30 = np.divide(flashfrequence * 100, npixels_30)
    flash_40 = np.divide(flashfrequence * 100, npixels_40)
    maxht20_minux_maxht40 = maxht20 - maxht40
    ellip_20 = np.divide(maxht20, npixels_20_R)
    ellip_30 = np.divide(maxht30, npixels_30_R)
    ellip_40 = np.divide(maxht40, npixels_40_R)
    ellip_20_maxht20 = np.multiply(np.divide(maxht20, npixels_20_R), maxht20)
    ellip_30_maxht30 = np.multiply(np.divide(maxht30, npixels_30_R), maxht30)
    ellip_40_maxht40 = np.multiply(np.divide(maxht40, npixels_40_R), maxht40)
    # 将所有的rainconv数据每一项除以volrain变成新的数据r数组
    r = np.divide(rainconv, volrain)
    index_add1 = list(np.where(r >= 0))
    index_add2 = list(np.where(minir > 0))
    index_add = list(set(index_add1[0]) & set(index_add2[0]))
    """add"""
    r = r[index_add]
    maxht20 = maxht20[index_add]
    maxht30 = maxht30[index_add]
    maxht40 = maxht40[index_add]
    npixels_20 = npixels_20[index_add]
    npixels_30 = npixels_30[index_add]
    npixels_40 = npixels_40[index_add]
    flash_20 = flash_20[index_add]
    flash_30 = flash_30[index_add]
    flash_40 = flash_40[index_add]
    minir = minir[index_add]
    flashrate = flashfrequence[index_add]
    maxht20_minux_maxht40 = maxht20_minux_maxht40[index_add]
    ellip_20 = ellip_20[index_add]
    ellip_30 = ellip_30[index_add]
    ellip_40 = ellip_40[index_add]
    ellip_20_maxht20 = ellip_20_maxht20[index_add]
    ellip_30_maxht30 = ellip_30_maxht30[index_add]
    ellip_40_maxht40 = ellip_40_maxht40[index_add]
    maxdbz = maxdbz[index_add]
    maxht = maxht[index_add]
    npixels_20_R = npixels_20_R[index_add]
    npixels_30_R = npixels_30_R[index_add]
    npixels_40_R = npixels_40_R[index_add]
    npx40_divide_npx20 = np.divide(npixels_40_R, npixels_20_R)
    npx40_divide_npx30 = np.divide(npixels_40_R, npixels_30_R)
    Volume20 = n20_volume[index_add]
    Volume40 = n40_volume[index_add]
    return maxht20, maxht30, maxht40, flashrate, npixels_20, npixels_30, npixels_40, flash_40, r, flash_20, flash_30, flash_40, minir,\
        flashrate, maxht20_minux_maxht40, ellip_20, ellip_30, ellip_40, maxdbz,\
        maxht, npx40_divide_npx20, npx40_divide_npx30, ellip_20_maxht20, ellip_30_maxht30, ellip_40_maxht40, Volume20, Volume40
# n20dbz, n30dbz, n40dbz, maxdbz


def for_(data, core_samples, labels, num=None, delete=None):
    mid = []
    if num==None:
        for i in data:
            i_mid = i[core_samples]
            mid.append(i_mid[np.where(labels != delete)])
    else:
        for i in data:
            i_mid = i[core_samples]
            mid.append(i_mid[np.where(labels == num)])
    return mid


def duqu_excel_julei(path):
    import pandas as pd
    file_path = path
    df = pd.read_excel(file_path, usecols=[1], names=None)
    dali = df.values.tolist()
    result = []
    for i in dali:
        result.append(i[0])
    return result


def ten_nine(x, former, latter):
    to_sort_data = sorted(x)
    index1 = list(np.where(x >= to_sort_data[round(len(to_sort_data) * former)]))
    index2 = list(np.where(x <= to_sort_data[round(len(to_sort_data) * latter)]))
    index = list(set(index1[0]) & set(index2[0]))
    return index


"""程序开始"""
# 程序发起点
plt.rc('axes', linewidth=3)
plt.tick_params(width=3)
name = ["Maxht20 (km)", "Maxht30 (km)", "Maxht40 (km)", r'FlRate (fl' + r'$\cdot$' + r'min$\mathregular{^{-1}}$)',
        r"Area20 (km$\mathregular{^{2}}$)",
        r"Area30 (km$\mathregular{^{2}}$)", r"Area40 (km$\mathregular{^{2}}$)",
        r'Fl40 (fl' + r'$\cdot$' + r'(100km)$\mathregular{^{-2}}$' +
                r'$\cdot$' + r'min$\mathregular{^{-1}}$)']
data = read_shuju()
# data_add = read_shuju_ADD()
# data_mid = []
# for i, j in zip(data, data_add):
#     data_mid.append(np.concatenate((i, j)))
# data = data_mid
core_samples = np.load("./core_samples.npy")
labels = np.load("./lables.npy")
print(labels)
print(len(labels))
data1 = for_(data, core_samples, labels, 3)
data2 = for_(data, core_samples, labels, 2)
data3 = for_(data, core_samples, labels, 1)
data = for_(data, core_samples, labels, None, 0)
# data4 = for_(data, stage4)
# data5 = for_(data, stage5)
# print(np.mean(data1[2] / (np.sqrt(data1[6] / math.pi) * 2)))
# print(np.mean(data2[2] / (np.sqrt(data2[6] / math.pi) * 2)))
# print(np.mean(data3[2] / (np.sqrt(data3[6] / math.pi) * 2)))
print(np.mean(data1[8]))
print(np.mean(data2[8]))
print(np.mean(data3[8]))
# print(np.mean(data4[7]))
# print(np.mean(data5[7]))
# print(len(data3[0]))
# print(len(data2[0]))
# print(len(data2[0]))
stage = ["All", "Pre-MT", "MT", "Post-MT"]
coefficient = 0
fig = plt.figure(coefficient, figsize=(9, 9))
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 35
mid = 1
for j in range(8, 9):
    ALL_STAGE = data[j]
    # ALL_STAGE = ALL_STAGE[ten_nine(ALL_STAGE, 0, 1)]
    # Development = data1[j]
    Pre_Maturity = data1[j]
    # Pre_Maturity = Pre_Maturity[ten_nine(Pre_Maturity, 0, 1)]
    Maturity = data2[j]
    # Maturity = Maturity[ten_nine(Maturity, 0, 1)]
    # Post_Maturity = data4[j]
    Dissipation = data3[j]
    # Dissipation = Dissipation[ten_nine(Dissipation, 0, 1)]
    ax = fig.add_subplot(1, 1, 1)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(direction='in', which="major", length=10, width=2,
                   top=False, right=False)
    # 用于设置y轴的刻度的大小即yticks的大小
    ax.tick_params(axis="y", labelsize=38)
    ax.minorticks_on()
    ax.tick_params(direction="in", which="minor", length=6, width=1.5,
                   top=False, right=False, bottom=False)
    ax.boxplot([ALL_STAGE, Pre_Maturity, Maturity, Dissipation], widths=0.4,
               showmeans=True, showfliers=False, whis=1, medianprops={"c": "black"},
               whiskerprops={"linestyle": "--", "linewidth": 2, "c": "black"})
    ax.set_xticklabels(stage)
    # ax.set_ylim(0, 1)
    # ax.set_xlabel(fontsize=40)
    # ax.set_yticks(fontsize=30)
    # ax.set_xticks(fontsize=30)
    ax.set_title(f"Ratio", fontsize=40)
    # ax.text(0.04, 0.9, abcdef[j], transform=ax.transAxes, fontsize=50)
    mid += 1
# plt.tight_layout(h_pad=0.5)
# plt.title("Different Stages of boxplot", fontsize=40)
# plt.yticks(fontsize=30)
# plt.xticks(fontsize=30)
plt.savefig(f"/root/git/Project_develop/figures/Ratio_boxplot.jpeg", bbox_inches="tight", dpi=400)
"""ax.text(
        0.2, 0.1, 'some text',
        horizontalalignment='center',  # 水平居中
        verticalalignment='center',  # 垂直居中
        transform=ax.transAxes  # 使用相对坐标
    )"""
