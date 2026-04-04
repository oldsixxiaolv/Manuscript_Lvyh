# -- coding:utf-8 --
from pyhdf.SD import SD, SDC
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FixedLocator
from scipy.signal import savgol_filter
import seaborn as sns
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.collections as mcollections


def duqu_excel():
    import pandas as pd
    # write later
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
    data = SD('../../TRMM_tropical_convection_dataset.hdf', SDC.READ)
    data_add = SD('../../TRMM_tropical_convection_dataset_202512.hdf', SDC.READ)
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
    # n20和n30和n40
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
    n30 = data_add.select("n30dbz")[:]
    n30 = n30[index]
    n30_mid = []
    for i, j in zip(n30, boost):
        if j == 1:
            mid = sum(i * 5 * 5 * 1.25)
        else:
            mid = sum(i * 4.3 * 4.3 * 1.25)
        n30_mid.append(mid)
    n30_volume = np.array(n30_mid)
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
    # mdbz
    mdbz = data.select("mdbz")[:]
    mdbz = mdbz[index]
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
    boost = boost[index_add]
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
    n20 = n20[index_add]
    n30 = n30[index_add]
    n40 = n40[index_add]
    mdbz = mdbz[index_add]
    return n20, n30, n40, flashrate, flash_40, boost


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


def chuli_mianji(n, boost):
    n_yes = []
    for i, j in zip(n, boost):
        if j == 1:
            n_yes.append(np.sqrt((i * 5 * 5) / math.pi))
        else:
            n_yes.append(np.sqrt((i * 4.3 * 4.3) / math.pi))
    return n_yes


def percent_value(Fl):
    rang = np.arange(0, 101, 5)
    Fl_mid_return = []
    for i in rang:
        if i == 0:
            index1 = np.where(Fl < np.percentile(Fl, 2.5))
            index2 = np.where(Fl >= np.percentile(Fl, 0))
            index = sorted(list(set(index1[0]) & set(index2[0])))
        elif i == 100:
            index1 = np.where(Fl < np.percentile(Fl, 100))
            index2 = np.where(Fl >= np.percentile(Fl, 97.5))
            index = sorted(list(set(index1[0]) & set(index2[0])))
        else:
            index1 = np.where(Fl < np.percentile(Fl, i+2.5))
            index2 = np.where(Fl >= np.percentile(Fl, i-2.5))
            index = sorted(list(set(index1[0]) & set(index2[0])))
        Fl_mid = np.mean(Fl[index])
        Fl_mid_return.append(Fl_mid)
    return Fl_mid_return


def plot_gradient_line(x, y, ax=None, cmap=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    if cmap is None:
        raise ValueError("A colormap must be provided.")
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors_for_cmap = np.linspace(0, 1, len(segments))
    lc = mcollections.LineCollection(segments, cmap=cmap, **kwargs)
    lc.set_array(colors_for_cmap)
    ax.add_collection(lc)
    ax.autoscale_view()
    return lc



"""程序开始"""
# 程序发起点
plt.rc('axes', linewidth=3)
plt.tick_params(width=3)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 50
data = read_shuju()
# data_add = read_shuju_ADD()
# data_mid = []
# for i, j in zip(data, data_add):
#     data_mid.append(np.concatenate((i, j)))
# data = data_mid
# stage = duqu_excel_julei(r"C:\Users\lvyih\Desktop\stage.xlsx")
# data_all = for_(data, stage)
core_samples = np.load("./core_samples.npy")
labels = np.load("./lables.npy")
data1 = for_(data, core_samples, labels, 3)
data2 = for_(data, core_samples, labels, 2)
data3 = for_(data, core_samples, labels, 1)
Height_n = np.arange(0, 20, 1.25)
window_length = 5
n20_develop = np.mean(chuli_mianji(data1[0], data1[5]), axis=0)
n20_develop = savgol_filter(n20_develop, window_length=window_length, polyorder=2)
n20_mature = np.mean(chuli_mianji(data2[0], data2[5]), axis=0)
n20_mature = savgol_filter(n20_mature, window_length=window_length, polyorder=2)
n20_dissi = np.mean(chuli_mianji(data3[0], data3[5]), axis=0)
n20_dissi = savgol_filter(n20_dissi, window_length=window_length, polyorder=2)
n30_develop = np.mean(chuli_mianji(data1[1], data1[5]), axis=0)
n30_develop = savgol_filter(n30_develop, window_length=window_length, polyorder=2)
n30_mature = np.mean(chuli_mianji(data2[1], data2[5]), axis=0)
n30_mature = savgol_filter(n30_mature, window_length=window_length, polyorder=2)
n30_dissi = np.mean(chuli_mianji(data3[1], data3[5]), axis=0)
n30_dissi = savgol_filter(n30_dissi, window_length=window_length, polyorder=2)
n40_develop = np.mean(chuli_mianji(data1[2], data1[5]), axis=0)
n40_develop = savgol_filter(n40_develop, window_length=window_length, polyorder=2)
n40_mature = np.mean(chuli_mianji(data2[2], data2[5]), axis=0)
n40_mature = savgol_filter(n40_mature, window_length=window_length, polyorder=2)
n40_dissi = np.mean(chuli_mianji(data3[2], data3[5]), axis=0)
n40_dissi = savgol_filter(n40_dissi, window_length=window_length, polyorder=2)
Flashrate_develop = data1[3]
FlRate_develop_percent = percent_value(Flashrate_develop)
# Flashrate_develop = savgol_filter(Flashrate_develop, window_length=window_length, polyorder=2)
Flashrate_mature = data2[3]
FlRate_mature_percent = percent_value(Flashrate_mature)
# Flashrate_mature = savgol_filter(Flashrate_mature, window_length=window_length, polyorder=2)
Flashrate_dissi = data3[3]
FlRate_dissi_percent = percent_value(Flashrate_dissi)
# Flashrate_dissi = savgol_filter(Flashrate_dissi, window_length=window_length, polyorder=2)
Flashrate_all = [FlRate_develop_percent, FlRate_mature_percent, FlRate_dissi_percent]
Fl40_develop = data1[4]
Fl40_develop_percent = percent_value(Fl40_develop)
# Fl40_develop = savgol_filter(Fl40_develop, window_length=window_length, polyorder=2)
Fl40_mature = data2[4]
Fl40_mature_percent = percent_value(Fl40_mature)
# Fl40_mature = savgol_filter(Fl40_mature, window_length=window_length, polyorder=2)
Fl40_dissi = data3[4]
Fl40_dissi_percent = percent_value(Fl40_dissi)
# Fl40_dissi = savgol_filter(Fl40_dissi, window_length=window_length, polyorder=2)
Fl40_all = [Fl40_develop_percent, Fl40_mature_percent, Fl40_dissi_percent]
# Flashrate_develop = np.mean(data1[3], axis=0)
# Flashrate_mature = np.mean(data2[3], axis=0)
# Flashrate_dissi = np.mean(data3[3], axis=0)
# Fl40_develop = np.mean(data1[4], axis=0)
# Fl40_mature = np.mean(data2[4], axis=0)
# Fl40_dissi = np.mean(data3[4], axis=0)
Middle = 0
Pre_mature = np.array([n20_develop, n30_develop, n40_develop])
max_pre_mature = max(n20_develop)
# max_pre_mature = [i*2, for i in [max(n20_develop), max(n30_develop), max(n40_develop)]]
Pre_mature_left = Middle - Pre_mature
Pre_mature_right = Pre_mature + Middle
Pre_left_right = [Pre_mature_left, Pre_mature_right]
mature = np.array([n20_mature, n30_mature, n40_mature])
max_mature = max(n20_mature)
mature_left = Middle - mature
mature_right = mature + Middle
mature_left_right = [mature_left, mature_right]
# max_mature = [i*2, for i in[max(n20_mature), max(n30_mature), max(n40_mature)]]
Post_mature = np.array([n20_dissi, n30_dissi, n40_dissi])
max_post_mature = max(n20_dissi)
Post_mature_left = Middle - Post_mature
Post_mature_right = Post_mature + Middle
Post_left_right = [Post_mature_left, Post_mature_right]
# max_post_mature = [i*2, for i in [max(n20_dissi), max(n30_dissi), max(n40_dissi)]]
All_left_right = [Pre_left_right, mature_left_right, Post_left_right]
_max_ = [max_pre_mature, max_mature, max_post_mature]





color = ["#4DAF4A", "#FFA500", "#E41A1C"]
labels = ["20 dBZ", "30 dBZ", "40 dBZ"]
# np.set_printoptions(threshold=np.inf)
fig = plt.figure(figsize=(40, 20))
gs = GridSpec(100, 108, figure=fig)
# abc = ["(a)", "(b)", "(c)"]
index = 0
for i in All_left_right:
    # 一定要记住的是tickparams中left是调整刻度线，labelleft才是调整刻度标签
    if index==0:
        ax = fig.add_subplot(gs[0:50, 0:32])
        ax_violin = fig.add_subplot(gs[62:100, 4:28])
        ax_violin2 = ax_violin.twinx()
        ax_violin.tick_params(axis="both", direction='out', length=10, width=2)
        ax_violin2.tick_params(axis="both", direction='out', length=10, width=2)
        ax.tick_params(axis="both", direction='out', length=10, width=2,
                    top=False, right=False)
        # 20dBZ
        ax.fill_betweenx(Height_n, i[0][0], i[1][0], hatch="|", color="#66C2A5", label="20 dBZ")
        # 30dBZ
        ax.fill_betweenx(Height_n, i[0][1], i[1][1], hatch="|", color="#FFC000", label="30 dBZ")
        # 40dBZ
        ax.fill_betweenx(Height_n, i[0][2], i[1][2], hatch="|", color="#B2182B", label="40 dBZ")
        ax.set_title("Pre-Mature", fontsize=50)
        ax.set_xlabel("Horizontal scale (km)", fontsize=50)
        ax.set_ylabel("Altitude (km)", fontsize=50, labelpad=1)
        ax.text(0.03, 0.85, "(a)", transform=ax.transAxes, fontsize=60)
        ax_violin.text(0.04, 0.83, "(b)", transform=ax_violin.transAxes, fontsize=60)
    elif index==1:
        ax = fig.add_subplot(gs[0:50, 38:70])
        ax_violin = fig.add_subplot(gs[62:100, 42:66])
        ax_violin2 = ax_violin.twinx()
        ax_violin.tick_params(axis="both", direction='out', length=10, width=2)
        ax_violin2.tick_params(axis="both", direction='out', length=10, width=2)
        ax.tick_params(axis="both", direction='out', length=10, width=2,
                    top=False, right=False, left=False, labelleft=False)
        # 20dBZ
        ax.fill_betweenx(Height_n, i[0][0], i[1][0], hatch="|", color="#66C2A5", label="20 dBZ")
        # 30dBZ
        ax.fill_betweenx(Height_n, i[0][1], i[1][1], hatch="|", color="#FFC000", label="30 dBZ")
        # 40dBZ
        ax.fill_betweenx(Height_n, i[0][2], i[1][2], hatch="|", color="#B2182B", label="40 dBZ")
        ax.set_title("Mature", fontsize=50)
        ax.set_xlabel("Horizontal scale (km)", fontsize=50)
        # ax.set_ylabel("Altitude", fontsize=50, labelpad=1)
        # ax.text(0.03, 0.85, "(b)", transform=ax.transAxes, fontsize=60)
        ax_violin.text(0.04, 0.83, "(c)", transform=ax_violin.transAxes, fontsize=60)
    else:
        ax = fig.add_subplot(gs[0:50, 76:108])
        ax_violin = fig.add_subplot(gs[62:100, 80:104])
        ax_violin2 = ax_violin.twinx()
        ax_violin.tick_params(axis="both", direction='out', length=10, width=2)
        ax_violin2.tick_params(axis="both", direction='out', length=10, width=2)
        ax.tick_params(axis="both", direction='out', length=10, width=2,
                    top=False, right=False, left=False, labelleft=False)
        # 20dBZ
        ax.fill_betweenx(Height_n, i[0][0], i[1][0], hatch="|", color="#66C2A5", label="20 dBZ")
        # 30dBZ
        ax.fill_betweenx(Height_n, i[0][1], i[1][1], hatch="|", color="#FFC000", label="30 dBZ")
        # 40dBZ
        ax.fill_betweenx(Height_n, i[0][2], i[1][2], hatch="|", color="#B2182B", label="40 dBZ")
        ax.legend(fontsize=35)
        ax.set_title("Post-Mature", fontsize=50)
        ax.set_xlabel("Horizontal scale (km)", fontsize=50)
        # ax.set_ylabel("Altitude", fontsize=50, labelpad=1)
        # ax.text(0.03, 0.85, "(c)", transform=ax.transAxes, fontsize=60)
        ax_violin.text(0.04, 0.83, "(d)", transform=ax_violin.transAxes, fontsize=60)
    # for w in i:
    #     baby = 0
    #     for q in w:
    #         ax.plot(q, Height_n, color[baby], label=labels[baby], linewidth=4.33)
    #         baby += 1    
    ax.xaxis.set_major_locator(FixedLocator(np.arange(-25, 25.5, 5)))
    xticklabel = []
    for i in np.arange(-25, 25.5, 5):
        if i%10==0:
            xticklabel.append(int(i))
        else:
            xticklabel.append("")
    ax.set_xticklabels(xticklabel)
    # ax.set_xticks(np.arange(0, 45, 5))
    ax.set_xlim(-27, 27)
    ax.set_ylim(-1, 21)
    # ax.text(0.05, 0.9, f"{abc[i]}", transform=ax.transAxes, fontsize=50)
    # ax.minorticks_on()
    # ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if index != 0:
        ax.spines["left"].set_visible(False)
    ax.grid(True)
    # if index == 1 or index == 2:
    #     ax.get_yaxis().set_visible(False)
    # ax.legend(fontsize=35)
    ###########################################
    ####### 绘制下面的小提琴图
    x = range(0, 101, 5)
    values = np.arange(0, 1.01, 0.05)
    # colors1 = ["#007370", "#007FA9", "#0086D8", "#0085F5", "#817BF8", "#DB68E1"]
    # colors2 = ["#008A8C", "#008A4C", "#498500", "#917500", "#CA5700", "#EC2853"]
    colors1 = ["#B5B5F4", "#9B9BF7", "#7C7CF5", "#203DF0", "#0002FF"]
    colors2 = ["#7C7CF5", "#F5939D", "#F77583", "#F52D4B", "#F9001C"]
    cmap1 = LinearSegmentedColormap.from_list('custom_cmap', colors1)
    cmap2 = LinearSegmentedColormap.from_list('custom_cmap', colors2)
    # cmap1 = plt.cm.get_cmap('')
    # cmap2 = plt.cm.get_cmap('')
    violin = ax_violin.scatter(x, Flashrate_all[index], s=160, alpha=1, c=values, cmap=cmap1, marker="o")
    plot_gradient_line(x, Flashrate_all[index], ax=ax_violin, cmap=cmap1, linewidth=4)
    #ax_violin.plot(x, Flashrate_all[index], color=cmap1(values))
    violin2 = ax_violin2.scatter(x, Fl40_all[index], s=250, alpha=1, c=values, cmap=cmap2, marker="*")
    plot_gradient_line(x, Fl40_all[index], ax=ax_violin2, cmap=cmap2, linewidth=4)
    #ax_violin2.plot(x, Fl40_all[index], color=cmap2(values))
    # ax_violin.violinplot([Flashrate_all[index], Fl40_all[index]])
    ax_violin.set_ylim(-5, 70.25)
    ax_violin.set_yticks([0, 15, 30, 45, 60])
    ax_violin2.set_ylim(-1.5, 26.2)
    ax_violin2.set_yticks([0, 6, 12, 18, 24])
    # ax_violin.spines["top"].set_visible(False)
    # ax_violin2.spines["top"].set_visible(False)
    ax_violin.tick_params(axis='y', labelcolor='#203DF0')
    ax_violin2.tick_params(axis='y', labelcolor='#F52D4B')
    # ax_violin.tick_params(axis='x', labelcolor='skyblue')
    # ax_violin2.tick_params(axis='x', labelcolor='lightgreen')
    # ax_violin.spines["right"].set_visible(False)
    # if index != 0:
    #     ax_violin.spines["left"].set_visible(False)
    # ax_xlabels = ax_violin.set_xticklabels(["FlRate", "FD$_{40}$"])
    # colors = ['#C0C9E4', '#C49D96']
    # for text, c in zip(ax_xlabels, colors):
    #     text.set_color(c)
    # ax_violin.set_ylabel(r'fl' + r'$\cdot$' + r'min$\mathregular{^{-1}}$', fontsize=40)
    # ax_violin2.set_ylabel(r'fl' + r'$\cdot$' + r'min$\mathregular{^{-1}}$' + r'$\cdot$' + r'(100km)$\mathregular{^{-2}}$', fontsize=40)
    ax_violin.set_ylabel(r'FlRate (fl' + r'$\cdot$' + r'min$\mathregular{^{-1}}$)', color="#203DF0", labelpad=10, fontsize=40)
    ax_violin2.set_ylabel(r'FD$_{40}$ (fl' + r'$\cdot$' +  r'min$\mathregular{^{-1}}$' + r'$\cdot$' + r'(100km)$\mathregular{^{-2}}$)', color="#F52D4B", labelpad=10, fontsize=40)
    # 为双y轴创建colorbar
    # cbar1 = fig.colorbar(violin, ax=ax_violin, pad=0.1) # pad 增加与图的间距
    # cbar1.set_label('FlRate', fontsize=40)
    # ax_pos = ax_violin.get_position()
    # cbar_ax = fig.add_axes([ax_pos.x1 + 2, ax_pos.y0, 0.02, ax_pos.height])
    # cbar2 = fig.colorbar(violin2, cax=cbar_ax) 
    # cbar2.set_label('FD$_{40}$', fontsize=10)


    ax_violin.set_xticks(range(0, 101, 5))
    xtickslabels = []
    for i in range(0, 101, 5):
        if i%20==0:
            xtickslabels.append(i)
        else:
            xtickslabels.append("")
    ax_violin.set_xticklabels(xtickslabels)
    ax_violin.set_xlabel("Percentage (%)", fontsize=50)
    # ax_violin.grid(True)
    # ax.set_ylabel("Altitude", fontsize=45)
    # set_axis_style(ax_violin, ["Flashrate", "FD$_{40}$"])
    index += 1
# n40_all = data[1]

# n40_develop = data1[1]

# n40_mature = data2[1]

# n40_dissi = data3[1]

# plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.size"] = 30
# fig = plt.figure(1, figsize=(30, 15))
# abcdef = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
# plt.tight_layout()
# plt.title("Different Stages of boxplot", fontsize=40)
# plt.yticks(fontsize=30)
# plt.xticks(fontsize=30)
plt.savefig(f"../../figures/fig10_violin_newest.jpeg", bbox_inches="tight", dpi=400)
"""ax.text(
        0.2, 0.1, 'some text',
        horizontalalignment='center',  # 水平居中
        verticalalignment='center',  # 垂直居中
        transform=ax.transAxes  # 使用相对坐标
    )"""
