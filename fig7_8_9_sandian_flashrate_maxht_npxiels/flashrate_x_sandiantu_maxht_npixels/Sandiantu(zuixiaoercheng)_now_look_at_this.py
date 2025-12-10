# -- coding:utf-8 --
from pyhdf.SD import SD, SDC
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
import statsmodels.api as sm
from scipy.interpolate import griddata
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.font_manager import FontProperties
from pandas import DataFrame
import math


def duqu_excel_julei(path):
    import pandas as pd
    file_path = path
    df = pd.read_excel(file_path, usecols=[1], names=None)
    dali = df.values.tolist()
    result = []
    for i in dali:
        result.append(i[0])
    return result


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
    n40 = n40[index_add]
    n20_volume = n20_volume[index_add]
    n40_volume = n40_volume[index_add]
    mdbz = mdbz[index_add]
    return flashrate, maxht20, maxht30, maxht40, maxdbz, minir, npixels_20_R, npixels_30_R, npixels_40_R, n20_volume, n40_volume


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


def Ex(x, a, b, c):
    return a * np.exp(b * x) + c


def _contour_(ax, x, y, levels):
    # xi = np.linspace(0, 220, 100)
    # yi = np.linspace(0, 20, 100)
    # xi, yi = np.meshgrid(xi, yi)
    # zi = griddata((x, y), z, (xi, yi), method="linear")
    # axx = ax.contour(xi, yi, zi, colors="black", extend="both", linewidths=3.5, levels=10)
    # plt.clabel(axx, inline=True, colors="black", fontsize=37)
    # 下面我们用核密度估计法对其进行操作
    # 核密度估计
    xy = np.vstack([x, y])  # 形状为(2, n)的数据
    kde = gaussian_kde(xy)
    # percentiles = [90, 95, 99, 99.5]
    # levels = [np.percentile(kde(np.vstack([x, y])), 100 - p) for p in percentiles]
    x_grid = np.linspace(x.min(), x.max(), 100)
    y_grid = np.linspace(y.min(), y.max(), 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_coords = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(grid_coords).reshape(X.shape)
    # 将Z进行归一化
    Z = (Z - Z.min())*100 / (Z.max() - Z.min())
    # 下面将核密度转成百分比
    # Percent = Z.reshape(-1)
    # levels = [np.percentile(Percent, p) for p in percentiles] 
    levels = [0.1, 1, 5, 10, 30, 50, 80]
    print(levels)
    # levels = [np.percentile(Z, 100 - p) for p in percentiles[::-1]]
    axx = ax.contour(X, Y, Z, colors="black", extend="both", linewidths=3.5, levels=levels)
    # font = FontProperties(weight='bold')
    def fmt(x):
        return f"{x:.1f}%" if x-0.1<0.0001 else f"{x:.0f}%"
    labels = plt.clabel(axx, inline=True, colors="black", fontsize=37, fmt=fmt)
    # 把所有 Text 对象加粗
    for txt in labels:
        txt.set_fontweight('bold')   # 或者 txt.set_weight('bold')
    # print("begin")
    # for i, col in enumerate(axx.collections):
        # 每条线的顶点
        # path = col.get_paths()[0]        # 第一条（最外）即可
        # x_mid, y_mid = path.vertices[len(path.vertices)//2]  # 中间点
        # plt.text(x_mid, y_mid, percentiles[i],
                # ha='center', va='center',
                # bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.8))


def _contour_change_ylim(ax, x, y, levels):
    x_2 = np.linspace(0, 220, 30)
    y_2 = np.linspace(y.min(), y.max(), 60)
    x_minus = x_2[1] - x_2[0]
    y_minus = y_2[1] - y_2[0]
    all_scale = []
    hh = 0
    for u in y_2:
        two_scale = []
        for v in x_2:
            indexx1 = list(np.where(y >= u))
            indexx2 = list(np.where(y < u + y_minus))
            indexxx1 = list(np.where(x >= v))
            indexxx2 = list(np.where(x < v + x_minus))
            indexq = list(set(indexx1[0]) & set(indexx2[0])
                          & set(indexxx1[0]) & set(indexxx2[0]))
            two_scale.append(len(indexq))
        all_scale.append(two_scale)
        print(hh)
        hh += 1
    axx = ax.contour(x_2, y_2, all_scale, colors="black", extend="both", linewidths=3.5, levels=levels)
    plt.clabel(axx, inline=True, colors="black", fontsize=37)


def for_j_and_mid(name_data, color_bar, ax, s):
    for kkk, www in zip(name_data[3:0:-1], color_bar[3:0:-1]):
        flashrate = kkk[0]
        parameter = kkk[j + 1]
        # 用4次多项式拟合
        x_mid = flashrate
        log_x = np.log(x_mid)
        y_mid = parameter
        ax.scatter(x_mid, y_mid, color=www, s=s)  # label='hetu_original_boxplot values'


"""程序开始"""
# 程序发起点
name = ["Flash", "Maxht20", "Maxht30", "Maxht40", "Maxdbz",
        "Minir", "R$_{eq}$20", "R$_{eq}$30", "R$_{eq}$40", "Volume20", "Volume40"]
data = read_shuju()
for i in data:
    print(len(i))
# data_add = read_shuju_ADD()
# data_mid = []
# for i, j in zip(data, data_add):
#     data_mid.append(np.concatenate((i, j)))
# data = data_mid
# stage4 = duqu_excel_julei(r"C:\Users\lvyih\Desktop\stage4.xlsx")
core_samples = np.load("./core_samples.npy")
print(len(core_samples))
labels = np.load("./lables.npy")
data1 = for_(data, core_samples, labels, 3)
data2 = for_(data, core_samples, labels, 2)
data3 = for_(data, core_samples, labels, 1)
data4 = for_(data, core_samples, labels, 0)
data = for_(data, core_samples, labels, None, 0)
print(len(data))
# data_all = data
# data1 = DataFrame(for_(data_all, stage1)).T
# data2 = DataFrame(for_(data_all, stage2)).T
# data3 = DataFrame(for_(data_all, stage3)).T
# data_all = DataFrame(data_all).T
# data_all = data_all.dropna(axis=0)
# data_all = np.array(data_all[data_all[0] > 0].T)
# data1 = data1.dropna(axis=0)
# data1 = np.array(data1[data1[0] > 0].T)
# data2 = data2.dropna(axis=0)
# data2 = np.array(data2[data2[0] > 0].T)
# data3 = data3.dropna(axis=0)
# data3 = np.array(data3[data3[0] > 0].T)
# data_add = read_shuju_ADD()
# data_mid = []
# for i, j in zip(data3, data_add):
#     data_mid.append(np.concatenate((i, j)))
# data3 = data_mid
# print(np.mean(data1[0]))
# print(np.mean(data2[0]))
# print(np.mean(data3[0]))
plt.rcParams['lines.linewidth'] = 3
name_data = [data, data1, data2, data3]
stage = ["All Stage", "Pre-Mature Stage", "Mature Stage", "Post-Mature Stage"]
coefficient = 0
abcd = ["(a)", "(b)", "(c)", "(d)"]
for j in range(7, 8):
    mid = 1
    for i in name_data:
        fig = plt.figure(coefficient, figsize=(30, 20))
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 40
        plt.rc('axes', linewidth=3)
        flashrate = i[0]
        parameter = i[j+1]
        # 确保绘图的volume量不能是零，从而避免拟合方程出现NAN的情况。
        if j == 9:
            index_40volume_0 = list(np.where(parameter != 0))
            index_40volu_0 = list(set(index_40volume_0[0]))
            flashrate = flashrate[index_40volu_0]
            parameter = parameter[index_40volu_0]
        # 用4次多项式拟合
        if j < 8:
            x = flashrate
            log_x = np.log(x)
            y = parameter
            # 创建对象ax1，ax2
            ax = fig.add_subplot(2, 2, mid)
            z1 = np.polyfit(log_x, y, 1)
            slope, intercept = z1[0], z1[1]
            correlation = np.corrcoef(log_x, y)[0, 1]
            equation_text = f'equation: y = {slope:.2f} * lnx + {intercept:.2f}'
            correlation_text = f'r = {correlation:.2f}'
            ax.tick_params(direction='in', which="major", length=15, width=3, pad=8)
            # 这个是专门针对x轴log变换的给画拟合线的代码
            if mid == 1:
                color_line = "black"
            elif mid == 2:
                color_line = "black"
            elif mid == 3:
                color_line = "black"
            elif mid == 4:
                color_line = "black"
            # ax.semilogx(np.unique(x), np.polyval(z1, np.unique(log_x)), color_line, linewidth=5)  # label='Fit'
            ax.semilogx(x, np.polyval(z1, log_x), color_line, linewidth=5)  # label='Fit'
            ax.set_xscale('log')
        else:
            x = flashrate
            log_x = np.log(x)
            y = parameter
            log_y = np.log(y)
            ax = fig.add_subplot(2, 2, mid)
            z1 = np.polyfit(log_x, log_y, 1)
            slope, intercept = z1[0], z1[1]
            correlation = np.corrcoef(log_x, log_y)[0, 1]
            equation_text = f'equation: lny = {slope:.2f} * lnx + {intercept:.2f}'
            correlation_text = f'r = {correlation:.2f}'
            ax.tick_params(direction='in', which="major", length=15, width=3, pad=8)
            # 以下是两个轴都进行变换的线性拟合代码
            if mid == 1:
                color_line = "black"
            elif mid == 2:
                color_line = "black"
            elif mid == 3:
                color_line = "black"
            elif mid == 4:
                color_line = "black"
            # ax.loglog(np.unique(x), np.exp(np.polyval(z1, np.unique(log_x))), color_line, linewidth=5)  # label='Fit'
            ax.loglog(x, np.exp(np.polyval(z1, log_x)), color_line, linewidth=5)  # label='Fit'
            ax.set_xscale('log')
            ax.set_yscale('log')
        ax.set_xlabel('FlRate (fl' + r'$\cdot$' +r'min$\mathregular{^{-1}}$)', fontsize=40)
        ax.set_xlim(0.4, 300)
        plt.yticks(fontsize=40)
        plt.xticks(fontsize=40)
        color_bar = ["black", "g", "red", "blue"]
        if j <= 2:
            ax.set_ylabel(f"{name[j+1]} (km)", fontsize=40)
        elif j == 3:
            # ax.set_ylabel(f"{name[j+1]}/km" + "$^{2}$", fontsize=40)
            ax.set_ylabel(f"{name[j + 1]} (dBZ)", fontsize=40)
        elif j == 4:
            ax.set_ylabel(f"{name[j + 1]} (K)", fontsize=40)
        elif j == 5:
            ax.set_ylabel(f"{name[j + 1]} (km)", fontsize=40)
        elif j == 6:
            ax.set_ylabel(f"{name[j + 1]} (km)", fontsize=40)
        elif j == 7:
            ax.set_ylabel(f"{name[j + 1]} (km)", fontsize=40)
        elif j == 8:
            ax.set_ylabel(f"{name[j + 1]}" + "(km$\mathregular{^{3}}$)", fontsize=40)
        elif j == 9:
            ax.set_ylabel(f"{name[j + 1]}" + "(km$\mathregular{^{3}}$)", fontsize=40)
        # plt.legend() # 指定legend的位置,读者可以自己help它的用法
        ax.set_title(f"{stage[mid-1]}", fontsize=40)
        # Maturity stage
        # Development stage
        # Dissipation stage
        if j == 0:
            ax.text(0.8, 0.1, f"b = {round(intercept, 2)}", transform=ax.transAxes, fontsize=40)
            ax.text(0.8, 0.2, f"k = {round(slope, 2)}", transform=ax.transAxes, fontsize=40)
            ax.text(0.8, 0.3, correlation_text, transform=ax.transAxes, fontsize=40)
        elif j == 1:
            ax.text(0.3, 0.15, equation_text, transform=ax.transAxes, fontsize=40)
            ax.text(0.3, 0.05, correlation_text, transform=ax.transAxes, fontsize=40)
        elif j == 2:
            # ax.text(0.3, 0.9, equation_text, transform=ax.transAxes, fontsize=40)
            ax.text(0.8, 0.1, correlation_text, transform=ax.transAxes, fontsize=40)
            ax.text(0.8, 0.2, f"b = {round(intercept, 2)}", transform=ax.transAxes, fontsize=40)
            ax.text(0.8, 0.3, f"k = {round(slope, 2)}", transform=ax.transAxes, fontsize=40)
        elif j == 3:
            ax.text(0.3, 0.15, equation_text, transform=ax.transAxes, fontsize=40)
            ax.text(0.3, 0.05, correlation_text, transform=ax.transAxes, fontsize=40)
        elif j == 4:
            ax.text(0.3, 0.9, equation_text, transform=ax.transAxes, fontsize=40)
            ax.text(0.3, 0.8, correlation_text, transform=ax.transAxes, fontsize=40)
        elif j == 5:
            ax.text(0.3, 0.15, equation_text, transform=ax.transAxes, fontsize=40)
            ax.text(0.3, 0.05, correlation_text, transform=ax.transAxes, fontsize=40)
        elif j == 6:
            ax.text(0.3, 0.15, equation_text, transform=ax.transAxes, fontsize=40)
            ax.text(0.3, 0.05, correlation_text, transform=ax.transAxes, fontsize=40)
        elif j == 7:
            ax.text(0.8, 0.1, correlation_text, transform=ax.transAxes, fontsize=40)
            ax.text(0.8, 0.2, f"b = {round(intercept, 2)}", transform=ax.transAxes, fontsize=40)
            ax.text(0.8, 0.3, f"k = {round(slope, 2)}", transform=ax.transAxes, fontsize=40)
            ax.set_ylim(0, 53)
        elif j == 8:
            ax.text(0.3, 0.15, equation_text, transform=ax.transAxes, fontsize=40)
            ax.text(0.3, 0.05, correlation_text, transform=ax.transAxes, fontsize=40)
        elif j == 9:
            ax.text(0.8, 0.1, correlation_text, transform=ax.transAxes, fontsize=40)
            ax.text(0.8, 0.2, f"b = {round(intercept, 2)}", transform=ax.transAxes, fontsize=40)
            ax.text(0.8, 0.3, f"k = {round(slope, 2)}", transform=ax.transAxes, fontsize=40)
            # 这里进行了y的范围的设定
            ax.set_ylim(15, 20000)
        if j == 0 and mid == 1:
            for_j_and_mid(name_data, color_bar, ax, 4)
            ax.set_ylim(0, 21)
            ax.text(0.04, 0.9, abcd[mid - 1], transform=ax.transAxes, fontsize=50)
            levels = [10, 25, 50, 100, 500, 1500]
            _contour_(ax, x, y, levels)
            # ax.legend(loc="upper left", bbox_to_anchor=(0.45, 0.3), prop={"family": "Times New Roman", "size": 40})
        elif j == 0 and mid == 2:
            ax.scatter(x, y, color=color_bar[mid-1], s=4)  # label='hetu_original_boxplot values'
            ax.set_ylim(0, 21)
            ax.text(0.04, 0.9, abcd[mid - 1], transform=ax.transAxes, fontsize=50)
            levels = [10, 25, 50, 100, 300, 500]
            _contour_(ax, x, y, levels)
            # levels = [400, 800],
            # ax.legend(loc="upper left", bbox_to_anchor=(0.45, 0.3), prop={"family": "Times New Roman", "size": 40})
        elif j == 0 and mid == 3:
            ax.scatter(x, y, color=color_bar[mid-1], s=4)  # label='hetu_original_boxplot values'
            ax.set_ylim(0, 21)
            ax.text(0.04, 0.9, abcd[mid - 1], transform=ax.transAxes, fontsize=50)
            levels = [10, 25, 50, 150, 300]
            _contour_(ax, x, y, levels)
            # ax.legend(loc="upper left", bbox_to_anchor=(0.45, 0.3), prop={"family": "Times New Roman", "size": 40})
        elif j == 0 and mid == 4:
            ax.scatter(x, y, color=color_bar[mid-1], s=4)  # label='hetu_original_boxplot values'
            ax.set_ylim(0, 21)
            ax.text(0.04, 0.9, abcd[mid - 1], transform=ax.transAxes, fontsize=50)
            levels = [10, 25, 50, 100, 500, 1000, 1500]
            _contour_(ax, x, y, levels)
            # ax.legend(loc="upper left", bbox_to_anchor=(0.45, 0.3), prop={"family": "Times New Roman", "size": 40})
        elif j == 1 and mid == 1:
            for_j_and_mid(name_data, color_bar, ax, 4)
            ax.set_ylim(0, 21)
            ax.text(0.04, 0.9, abcd[mid - 1], transform=ax.transAxes, fontsize=50)
            levels = [10, 25, 50, 300, 500, 1000, 3000]
            _contour_(ax, x, y, levels)
            # ax.legend(loc="upper left", bbox_to_anchor=(0.45, 0.3), prop={"family": "Times New Roman", "size": 40})
        elif j == 1 and mid == 2:
            ax.scatter(x, y, color=color_bar[mid-1], s=4)
            ax.set_ylim(0, 21)
            ax.text(0.04, 0.9, abcd[mid - 1], transform=ax.transAxes, fontsize=50)
            levels = [10, 25, 50, 300, 1000, 1600]
            _contour_(ax, x, y, levels)
            # ax.legend(loc="upper left", bbox_to_anchor=(0.45, 0.3), prop={"family": "Times New Roman", "size": 40})
        elif j == 1 and mid == 3:
            ax.scatter(x, y, color=color_bar[mid-1], s=4)
            ax.set_ylim(0, 21)
            ax.text(0.04, 0.9, abcd[mid - 1], transform=ax.transAxes, fontsize=50)
            levels = [10, 25, 50, 150, 300, 600]
            _contour_(ax, x, y, levels)
            # ax.legend(loc="upper left", bbox_to_anchor=(0.45, 0.3), prop={"family": "Times New Roman", "size": 40})
        elif j == 1 and mid == 4:
            ax.scatter(x, y, color=color_bar[mid-1], s=4)
            ax.set_ylim(0, 21)
            ax.text(0.04, 0.9, abcd[mid - 1], transform=ax.transAxes, fontsize=50)
            levels = [10, 25, 50, 100, 800, 1600, 3200]
            _contour_(ax, x, y, levels)
            # ax.legend(loc="upper left", bbox_to_anchor=(0.45, 0.3), prop={"family": "Times New Roman", "size": 40})
        elif j == 2 and mid == 1:
            for_j_and_mid(name_data, color_bar, ax, 4)
            ax.set_ylim(0, 21)
            ax.text(0.04, 0.9, abcd[mid - 1], transform=ax.transAxes, fontsize=50)
            levels = [50, 500, 1500, 3000]
            _contour_(ax, x, y, levels)
            # ax.legend(loc="upper left", bbox_to_anchor=(0.45, 1), prop={"family": "Times New Roman", "size": 40})
        elif j == 2 and mid == 2:
            ax.scatter(x, y, color=color_bar[mid-1], s=4)
            ax.set_ylim(0, 21)
            ax.text(0.04, 0.9, abcd[mid - 1], transform=ax.transAxes, fontsize=50)
            levels = [50, 300, 1000, 1600]
            _contour_(ax, x, y, levels)
            # ax.legend(loc="upper left", bbox_to_anchor=(0.45, 1), prop={"family": "Times New Roman", "size": 40})
        elif j == 2 and mid == 3:
            ax.scatter(x, y, color=color_bar[mid-1], s=4)
            ax.set_ylim(0, 21)
            ax.text(0.04, 0.9, abcd[mid - 1], transform=ax.transAxes, fontsize=50)
            levels = [50, 150, 300, 600]
            _contour_(ax, x, y, levels)
            # ax.legend(loc="upper left", bbox_to_anchor=(0.45, 1), prop={"family": "Times New Roman", "size": 40})
        elif j == 2 and mid == 4:
            ax.scatter(x, y, color=color_bar[mid-1], s=4)
            ax.set_ylim(0, 21)
            ax.text(0.04, 0.9, abcd[mid - 1], transform=ax.transAxes, fontsize=50)
            levels = [50, 800, 1600, 3200]
            _contour_(ax, x, y, levels)
            # ax.legend(loc="upper left", bbox_to_anchor=(0.45, 1), prop={"family": "Times New Roman", "size": 40})
        elif j == 3 and mid == 1:
            for_j_and_mid(name_data, color_bar, ax, 4)
            ax.set_ylim(30, 65)
            ax.text(0.04, 0.9, abcd[mid - 1], transform=ax.transAxes, fontsize=50)
            levels = [10, 25, 50, 500, 1500, 3000]
            _contour_(ax, x, y, levels)
            # ax.legend(loc="upper left", bbox_to_anchor=(0.45, 0.3), prop={"family": "Times New Roman", "size": 40})
        elif j == 3 and mid == 2:
            ax.scatter(x, y, color=color_bar[mid - 1], s=4)  # label='hetu_original_boxplot values'
            ax.set_ylim(30, 65)
            ax.text(0.04, 0.9, abcd[mid - 1], transform=ax.transAxes, fontsize=50)
            levels = [10, 25, 50, 300, 1000, 1600]
            _contour_(ax, x, y, levels)
            # ax.legend(loc="upper left", bbox_to_anchor=(0.45, 0.3), prop={"family": "Times New Roman", "size": 40})
        elif j == 3 and mid == 3:
            ax.scatter(x, y, color=color_bar[mid - 1], s=4)  # label='hetu_original_boxplot values'
            ax.set_ylim(30, 65)
            ax.text(0.04, 0.9, abcd[mid - 1], transform=ax.transAxes, fontsize=50)
            levels = [10, 25, 50, 150, 300, 600]
            _contour_(ax, x, y, levels)
            # ax.legend(loc="upper left", bbox_to_anchor=(0.45, 0.3), prop={"family": "Times New Roman", "size": 40})
        elif j == 3 and mid == 4:
            ax.scatter(x, y, color=color_bar[mid - 1], s=4)  # label='hetu_original_boxplot values'
            ax.set_ylim(30, 65)
            ax.text(0.04, 0.9, abcd[mid - 1], transform=ax.transAxes, fontsize=50)
            levels = [10, 25, 50, 100, 800, 1600, 3200]
            _contour_(ax, x, y, levels)
            # ax.legend(loc="upper left", bbox_to_anchor=(0.45, 0.3), prop={"family": "Times New Roman", "size": 40})
        elif j == 4 and mid == 1:
            for_j_and_mid(name_data, color_bar, ax, 4)
            # ax.legend(loc="upper left", bbox_to_anchor=(0.4, 1), prop={"family": "Times New Roman", "size": 40})
            ax.set_ylim(160, 280)
            ax.text(0.04, 0.9, abcd[mid - 1], transform=ax.transAxes, fontsize=50)
            levels = [10, 25, 50, 500, 1500, 3000]
            _contour_(ax, x, y, levels)
        elif j == 4 and mid == 2:
            ax.scatter(x, y, color=color_bar[mid - 1], s=4)  # label='hetu_original_boxplot values'
            # ax.legend(loc="upper left", bbox_to_anchor=(0.4, 1), prop={"family": "Times New Roman", "size": 40})
            ax.set_ylim(160, 280)
            ax.text(0.04, 0.9, abcd[mid - 1], transform=ax.transAxes, fontsize=50)
            levels = [10, 25, 50, 300, 1000, 1600]
            _contour_(ax, x, y, levels)
        elif j == 4 and mid == 3:
            ax.scatter(x, y, color=color_bar[mid - 1], s=4)  # label='hetu_original_boxplot values'
            # ax.legend(loc="upper left", bbox_to_anchor=(0.4, 1), prop={"family": "Times New Roman", "size": 40})
            ax.set_ylim(160, 280)
            ax.text(0.04, 0.9, abcd[mid - 1], transform=ax.transAxes, fontsize=50)
            levels = [10, 25, 50, 150, 300, 600]
            _contour_(ax, x, y, levels)
        elif j == 4 and mid == 4:
            ax.scatter(x, y, color=color_bar[mid - 1], s=4)  # label='hetu_original_boxplot values'
            # ax.legend(loc="upper left", bbox_to_anchor=(0.4, 1), prop={"family": "Times New Roman", "size": 40})
            ax.set_ylim(160, 280)
            ax.text(0.04, 0.9, abcd[mid - 1], transform=ax.transAxes, fontsize=50)
            levels = [10, 25, 50, 800, 1600, 3200]
            _contour_(ax, x, y, levels)
        elif j == 5 and mid == 1:
            for_j_and_mid(name_data, color_bar, ax, 4)
            ax.text(0.04, 0.1, abcd[mid - 1], transform=ax.transAxes, fontsize=50)
            levels = [10, 25, 50, 100, 300]
            _contour_(ax, x, y, levels)
        elif j == 5 and mid == 2:
            ax.scatter(x, y, color=color_bar[mid - 1], s=2)
            ax.text(0.04, 0.1, abcd[mid - 1], transform=ax.transAxes, fontsize=50)
            levels = [10, 25, 50, 80, 140]
            _contour_(ax, x, y, levels)
        elif j == 5 and mid == 3:
            ax.scatter(x, y, color=color_bar[mid - 1], s=2)
            ax.text(0.04, 0.1, abcd[mid - 1], transform=ax.transAxes, fontsize=50)
            levels = [10, 20, 30, 40, 50]
            _contour_(ax, x, y, levels)
        elif j == 5 and mid == 4:
            ax.scatter(x, y, color=color_bar[mid - 1], s=2)
            ax.text(0.04, 0.1, abcd[mid - 1], transform=ax.transAxes, fontsize=50)
            levels = [10, 25, 50, 100, 150, 300]
            _contour_(ax, x, y, levels)
        elif j == 6 and mid == 1:
            for_j_and_mid(name_data, color_bar, ax, 4)
            ax.text(0.04, 0.9, abcd[mid - 1], transform=ax.transAxes, fontsize=50)
            levels = [50, 500, 1500, 3000]
            _contour_(ax, x, y, levels)
        elif j == 6 and mid == 2:
            ax.scatter(x, y, color=color_bar[mid - 1], s=2)
            ax.text(0.04, 0.9, abcd[mid - 1], transform=ax.transAxes, fontsize=50)
            levels = [50, 300, 1000, 1600]
            _contour_(ax, x, y, levels)
        elif j == 6 and mid == 3:
            ax.scatter(x, y, color=color_bar[mid - 1], s=2)
            ax.text(0.04, 0.9, abcd[mid - 1], transform=ax.transAxes, fontsize=50)
            levels = [50, 150, 300, 600]
            _contour_(ax, x, y, levels)
        elif j == 6 and mid == 4:
            ax.scatter(x, y, color=color_bar[mid - 1], s=2)
            ax.text(0.04, 0.9, abcd[mid - 1], transform=ax.transAxes, fontsize=50)
            levels = [50, 800, 1600, 3200]
            _contour_(ax, x, y, levels)
        elif j == 7 and mid == 1:
            for_j_and_mid(name_data, color_bar, ax, 4)
            ax.text(0.04, 0.9, abcd[mid - 1], transform=ax.transAxes, fontsize=50)
            levels = [50, 500, 1500, 3000]
            _contour_(ax, x, y, levels)
        elif j == 7 and mid == 2:
            ax.scatter(x, y, color=color_bar[mid - 1], s=2)
            ax.text(0.04, 0.9, abcd[mid - 1], transform=ax.transAxes, fontsize=50)
            levels = [50, 300, 1000, 1600]
            _contour_(ax, x, y, levels)
        elif j == 7 and mid == 3:
            ax.scatter(x, y, color=color_bar[mid - 1], s=2)
            ax.text(0.04, 0.9, abcd[mid - 1], transform=ax.transAxes, fontsize=50)
            levels = [50, 150, 300, 600]
            _contour_(ax, x, y, levels)
        elif j == 7 and mid == 4:
            ax.scatter(x, y, color=color_bar[mid - 1], s=2)
            ax.text(0.04, 0.9, abcd[mid - 1], transform=ax.transAxes, fontsize=50)
            levels = [50, 800, 1600, 3200]
            _contour_(ax, x, y, levels)
        elif j == 8 and mid == 1:
            for_j_and_mid(name_data, color_bar, ax, 4)
            ax.text(0.04, 0.9, abcd[mid - 1], transform=ax.transAxes, fontsize=50)
            levels = [50, 500, 1500, 3000]
            _contour_(ax, x, y, levels)
        elif j == 8 and mid == 2:
            ax.scatter(x, y, color=color_bar[mid - 1], s=2)
            ax.text(0.04, 0.9, abcd[mid - 1], transform=ax.transAxes, fontsize=50)
            levels = [50, 300, 1000, 1600]
            _contour_(ax, x, y, levels)
        elif j == 8 and mid == 3:
            ax.scatter(x, y, color=color_bar[mid - 1], s=2)
            ax.text(0.04, 0.9, abcd[mid - 1], transform=ax.transAxes, fontsize=50)
            levels = [50, 150, 300, 600]
            _contour_(ax, x, y, levels)
        elif j == 8 and mid == 4:
            ax.scatter(x, y, color=color_bar[mid - 1], s=2)
            ax.text(0.04, 0.9, abcd[mid - 1], transform=ax.transAxes, fontsize=50)
            levels = [50, 800, 1600, 3200]
            _contour_(ax, x, y, levels)
        elif j == 9 and mid == 1:
            for_j_and_mid(name_data, color_bar, ax, 4)
            ax.text(0.04, 0.9, abcd[mid - 1], transform=ax.transAxes, fontsize=50)
            levels = [50, 500, 1500, 3000]
            _contour_(ax, x, y, levels)
        elif j == 9 and mid == 2:
            ax.scatter(x, y, color=color_bar[mid - 1], s=2)
            ax.text(0.04, 0.9, abcd[mid - 1], transform=ax.transAxes, fontsize=50)
            levels = [50, 300, 1000, 1600]
            _contour_(ax, x, y, levels)
        elif j == 9 and mid == 3:
            ax.scatter(x, y, color=color_bar[mid - 1], s=2)
            ax.text(0.04, 0.9, abcd[mid - 1], transform=ax.transAxes, fontsize=50)
            levels = [50, 150, 300, 600]
            _contour_(ax, x, y, levels)
        elif j == 9 and mid == 4:
            ax.scatter(x, y, color=color_bar[mid - 1], s=2)
            ax.text(0.04, 0.9, abcd[mid - 1], transform=ax.transAxes, fontsize=50)
            levels = [50, 800, 1600, 3200]
            _contour_(ax, x, y, levels)
        mid += 1
    plt.tight_layout()
    plt.savefig(f"/root/git/Project_develop/figures/{name[j+1]}_all_stage.png", bbox_inches="tight",
                dpi=50)
    # plt.show()
    coefficient += 1
    # os.system("pause")
"""ax.text(
        0.2, 0.1, 'some text',
        horizontalalignment='center',  # 水平居中
        verticalalignment='center',  # 垂直居中
        transform=ax.transAxes  # 使用相对坐标
    )"""
