# -- coding:utf-8 --
from pyhdf.SD import SD, SDC
import numpy as np
import matplotlib.pyplot as plt
import math
import decimal


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
    file_path = r'E:\\Programing\\school_creation_python311\\' \
                'xiaochuang\\2024_thesis\\Excel_and_data_tropical_newest_no_executable\\' \
                'duqu_last.xlsx'
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
    filepath = r'E:\\Programing\\school_creation_python311\\' \
               'xiaochuang\\2024_thesis\\Excel_and_data_tropical_newest_no_executable\\' \
               'TRMM_tropical_convection_dataset.hdf'
    data = SD(filepath, SDC.READ)
    longitude = list(data.select("longitude")[:])
    latitude = list(data.select("latitude")[:])
    landocean = list(data.select("landocean")[:])
    longitude = np.array(list(map(int, longitude)))
    latitude = np.array(list(map(int, latitude)))
    landocean = np.array(list(map(int, landocean)))
    flashcount = data.select('flashcount')[:]
    flashcount = np.array(list(map(int, flashcount)))
    maxht20 = data.select("maxht20")[:]
    maxht20 = np.array(list(map(int, maxht20)))
    volrain = data.select('volrain')[:]
    rainconv = data.select('rainconv')[:]
    npixels_20 = data.select("npixels_20")[:]
    npixels_40 = data.select("npixels_40")[:]
    viewtime = data.select("viewtime")[:]
    maxht40 = data.select("maxht40")[:]
    maxht40 = np.array(list(map(int, maxht40)))
    # boost = data.select("boost")[:]
    # npixels_40 = data.select("npixels_40")[:]
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
    # 两个数组取交集
    index = list(set(index1[0]) & set(index2[0]) & set(index3[0]) & set(index4[0]) &
                 set(index5[0]) & set(index6[0]) & set(index7[0]) & set(index8[0]) &
                 set(index11[0]) & set(index12[0]) & set(index13[0]))
    index = sorted(index)
    indexw = duqu_excel()
    r = r[index]
    r = r[indexw]
    # 把hdf文件中的闪电频数数据提取出来
    flashcount = data.select('flashcount')[:]
    flashcount = flashcount[index]
    flashcount = flashcount[indexw]
    # 把hdf文件中总的降水数据提取出来
    volrain = data.select('volrain')[:]
    volrain = volrain[index]
    volrain = volrain[indexw]
    # 把hdf文件中的层云降水数据提取出来
    rainstrat = data.select('rainstrat')[:]
    rainstrat = rainstrat[index]
    rainstrat = rainstrat[indexw]
    # 把hdf文件中的对流降水数据分离出
    rainconv = data.select('rainconv')[:]
    rainconv = rainconv[index]
    rainconv = rainconv[indexw]
    # 把hdf文件中的观测时间数据提取出来
    viewtime = data.select('viewtime')[:]
    viewtime = viewtime[index]
    viewtime = viewtime[indexw]
    # 把hdf文件是否升轨数据提取出来
    boost = data.select("boost")[:]
    boost = boost[index]
    boost = boost[indexw]
    # 把hdf文件的最大顶高数据提取出来（这个数据老师说第一个maxht不用操作）
    maxht = data.select("maxht")[:]
    maxht = maxht[index]
    maxht = maxht[indexw]
    maxht20 = data.select("maxht20")[:]
    maxht20 = maxht20[index]
    maxht20 = maxht20[indexw]
    maxht30 = data .select("maxht30")[:]
    maxht30 = maxht30[index]
    maxht30 = maxht30[indexw]
    maxht40 = data.select("maxht40")[:]
    maxht40 = maxht40[index]
    maxht40 = maxht40[indexw]
    # 这个是云的顶层的亮温，可以通过知晓这个温度的高低来判断云顶的高度，温度越低意味着云越高
    minir = data.select("minir")[:]
    minir = minir[index]
    minir = minir[indexw]
    # 这个先不用
    npixels_40 = data.select("npixels_40")[:]
    npixels_40 = npixels_40[index]
    npixels_40 = npixels_40[indexw]
    npixels_40 = npixels_size(npixels_40, boost)
    npixels_40_K = np.multiply(npixels_40, 4)
    npixels_40_K = np.divide(npixels_40_K, math.pi)
    npixels_40_R = np.sqrt(npixels_40_K)
    npixels_30 = data.select("npixels_30")[:]
    npixels_30 = npixels_30[index]
    npixels_30 = npixels_30[indexw]
    npixels_30 = npixels_size(npixels_30, boost)
    npixels_30_K = np.multiply(npixels_30, 4)
    npixels_30_K = np.divide(npixels_30_K, math.pi)
    npixels_30_R = np.sqrt(npixels_30_K)
    npixels_20 = data.select("npixels_20")[:]
    npixels_20 = npixels_20[index]
    npixels_20 = npixels_20[indexw]
    npixels_20 = npixels_size(npixels_20, boost)
    npixels_20_K = np.multiply(npixels_20, 4)
    npixels_20_K = np.divide(npixels_20_K, math.pi)
    npixels_20_R = np.sqrt(npixels_20_K)
    maxnsrain = data.select("maxnsrain")[:]
    maxnsrain = maxnsrain[index]
    maxnsrain = maxnsrain[indexw]
    flashcount = flashcount.astype(float)
    flashrate = np.divide(flashcount, viewtime)
    flashrate = np.multiply(flashrate, 60)
    maxdbz = data.select("maxdbz")[:]
    maxdbz = maxdbz[index]
    maxdbz = maxdbz[indexw]
    n40 = data.select("n40dbz")[:]
    n40 = n40[index]
    n40 = n40[indexw]
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
    n20 = n20[indexw]
    n20_mid = []
    for i, j in zip(n20, boost):
        if j == 1:
            mid = sum(i * 5 * 5 * 1.25)
        else:
            mid = sum(i * 4.3 * 4.3 * 1.25)
        n20_mid.append(mid)
    n20_volume = np.array(n20_mid)
    inde = list(np.where(r > 0.67))
    index_add2 = list(np.where(minir > 0))
    inded = list(set(inde[0]) & set(index_add2[0]))
    flashrate = flashrate[inded]
    maxht40 = maxht40[inded]
    maxht20 = maxht20[inded]
    maxht30 = maxht30[inded]
    npixels_20_R = npixels_20_R[inded]
    npixels_30_R = npixels_30_R[inded]
    npixels_40_R = npixels_40_R[inded]
    npixels_20 = npixels_20[inded]
    npixels_30 = npixels_30[inded]
    npixels_40 = npixels_40[inded]
    ellip_20 = np.divide(maxht20 * 100, npixels_20_R)
    ellip_30 = np.divide(maxht30 * 100, npixels_30_R)
    ellip_40 = np.divide(maxht40 * 100, npixels_40_R)
    maxht = maxht[inded]
    maxdbz = maxdbz[inded]
    flashrate_20 = np.divide(flashrate * 100, npixels_20)
    flashrate_40 = np.divide(flashrate * 100, npixels_40)
    minir = minir[inded]
    n40_volume = n40_volume[inded]
    n20_volume = n20_volume[inded]
    return flashrate, maxht20, npixels_20, n20_volume


def for_(data, stage):
    mid = []
    for i in data:
        mid.append(i[stage])
    return mid


def Ex(x, a, b, c):
    return a * np.exp(b * x) + c


def _contour_(ax, x, y, x_max, y_max, x_minus, y_minus, levels):
    x_2 = np.arange(0, x_max, x_minus)
    y_2 = np.arange(0, y_max, y_minus)
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
    axx = ax.contour(x_2, y_2, all_scale, colors="black", extend="both", linewidths=5, levels=levels)
    plt.clabel(axx, inline=True, colors="black", fontsize=37)


def _contour_change_ylim(ax, x, y, x_max, y_min, y_max, x_minus, y_minus, levels):
    x_2 = np.arange(0, x_max, x_minus)
    y_2 = np.arange(y_min, y_max, y_minus)
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
    axx = ax.contour(x_2, y_2, all_scale, colors="black", extend="both", linewidths=5, levels=levels)
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
# 调整全局的精度
# decimal.getcontext().prec = 100
# 不限制输出数据的多少
np.set_printoptions(threshold=np.Inf)
# 程序发起点
name = ["Flashrate", "Maxht20", "Area20", "Volume20"]
data = read_shuju()
# data_add = read_shuju_ADD()
# data_mid = []
# for i, j in zip(data, data_add):
#     data_mid.append(np.concatenate((i, j)))
# data = data_mid
stage1 = duqu_excel_julei(r'E:\\Programing\\school_creation_python311\\' \
                          'xiaochuang\\2024_thesis\\Excel_and_data_tropical_newest_no_executable\\' \
                          'stage1.xlsx')
stage2 = duqu_excel_julei(r'E:\\Programing\\school_creation_python311\\' \
                          'xiaochuang\\2024_thesis\\Excel_and_data_tropical_newest_no_executable\\' \
                          'stage2.xlsx')
stage3 = duqu_excel_julei(r'E:\\Programing\\school_creation_python311\\' \
                          'xiaochuang\\2024_thesis\\Excel_and_data_tropical_newest_no_executable\\' \
                          'stage3.xlsx')
# stage4 = duqu_excel_julei(r"C:\Users\lvyih\Desktop\stage4.xlsx")
data1 = for_(data, stage1)
data2 = for_(data, stage2)
data3 = for_(data, stage3)
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
fig = plt.figure(coefficient, figsize=(36, 10))
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 30
plt.rc('axes', linewidth=3)
for j in range(0, 3):
    # 确保绘图的volume量不能是零，从而避免拟合方程出现NAN的情况。
    # 用4次多项式拟合
    if j == 0:
        ax = fig.add_subplot(1, 3, j + 1)
        kkkkk = 0
        for w in name_data:
            flashrate = w[0]
            parameter = w[j + 1]
            x = flashrate
            log_x = np.log(x)
            y = parameter
            # unique_number, counts = np.unique(x, return_counts=True)
            # duplicates = unique_number[counts > 1]
            # print(len(duplicates))
            # unique_2, count = np.unique(log_x, return_counts=True)
            # duplicates2 = unique_2[count > 1]
            # print(len(duplicates2))
            # 创建对象ax1，ax2
            z1 = np.polyfit(log_x, y, 1)
            slope, intercept = z1[0], z1[1]
            correlation = np.corrcoef(log_x, y)[0, 1]
            equation_text = f'equation: y = {slope:.2f} * lnx + {intercept:.2f}'
            correlation_text = f'r = {correlation:.2f}'
            if kkkkk == 0:
                # 因为x不同的有可能logx是相同的，所以要改一下
                ax.semilogx(np.unique(x), np.polyval(z1, np.log(np.unique(x))), 'black', linewidth=5)  # label='Fit'
                ax.set_xscale('log')
                ax.set_ylim(0, 21)
                ax.tick_params(direction='in', which="major", length=15, width=3, pad=8)
                ax.text(0.8, 0.4, correlation_text, transform=ax.transAxes, color="black", fontsize=40)
            elif kkkkk == 1:
                ax1 = ax.twinx()
                ax1.semilogx(np.unique(x), np.polyval(z1, np.log(np.unique(x))), 'darkgreen', linewidth=5)  # label='Fit'
                ax1.set_xscale('log')
                ax1.set_ylim(0, 21)
                ax1.axes.yaxis.set_visible(False)
                ax1.text(0.8, 0.3, correlation_text, transform=ax.transAxes, color="darkgreen", fontsize=40)
            elif kkkkk == 2:
                ax2 = ax1.twinx()
                ax2.semilogx(np.unique(x), np.polyval(z1, np.log(np.unique(x))), 'red', linewidth=5)  # label='Fit'
                ax2.set_xscale('log')
                ax2.set_ylim(0, 21)
                ax2.axes.yaxis.set_visible(False)
                ax.text(0.8, 0.2, correlation_text, transform=ax.transAxes, color="red", fontsize=40)
            elif kkkkk == 3:
                ax3 = ax2.twinx()
                ax3.semilogx(np.unique(x), np.polyval(z1, np.log(np.unique(x))), 'mediumblue', linewidth=5)  # label='Fit'
                ax3.set_xscale('log')
                ax3.set_ylim(0, 21)
                ax3.axes.yaxis.set_visible(False)
                ax.text(0.8, 0.1, correlation_text, transform=ax.transAxes, color="mediumblue", fontsize=40)
            kkkkk += 1
        # 这个是专门针对x轴log变换的给画拟合线的代码
    else:
        ax = fig.add_subplot(1, 3, j + 1)
        kkkk = 0
        for w in name_data:
            flashrate = w[0]
            parameter = w[j + 1]
            x = flashrate
            log_x = np.log(x)
            y = parameter
            log_y = np.log(y)
            z1 = np.polyfit(log_x, log_y, 1)
            slope, intercept = z1[0], z1[1]
            correlation = np.corrcoef(log_x, log_y)[0, 1]
            equation_text = f'equation: lny = {slope:.2f} * lnx + {intercept:.2f}'
            correlation_text = f'r = {correlation:.2f}'
            if kkkk == 0:
                ax.tick_params(direction='in', which="major", length=15, width=3, pad=8)
                # 以下是两个轴都进行变换的线性拟合代码
                ax.loglog(np.unique(x), np.exp(np.polyval(z1, np.log(np.unique(x)))), 'black', linewidth=5)  # label='Fit'
                ax.set_xscale('log')
                ax.set_yscale('log')
                if j == 1:
                    ax.set_ylim(20, 25000)
                elif j == 2:
                    ax.set_ylim(150, 250000)
                ax.text(0.8, 0.4, correlation_text, transform=ax.transAxes, color="black", fontsize=40)
            elif kkkk == 1:
                ax1 = ax.twinx()
                ax1.tick_params(direction='in', which="major", length=15, width=3, pad=8)
                # 以下是两个轴都进行变换的线性拟合代码
                ax1.loglog(np.unique(x), np.exp(np.polyval(z1, np.log(np.unique(x)))), 'darkgreen', linewidth=5)  # label='Fit'
                ax1.set_xscale('log')
                ax1.set_yscale('log')
                if j == 1:
                    ax1.set_ylim(20, 25000)
                elif j == 2:
                    ax1.set_ylim(150, 250000)
                ax1.axes.yaxis.set_visible(False)
                ax.text(0.8, 0.3, correlation_text, transform=ax.transAxes, color="darkgreen", fontsize=40)
            elif kkkk == 2:
                ax2 = ax1.twinx()
                ax2.tick_params(direction='in', which="major", length=15, width=3, pad=8)
                # 以下是两个轴s都进行变换的线性拟合代码
                ax2.loglog(np.unique(x), np.exp(np.polyval(z1, np.log(np.unique(x)))), 'red', linewidth=5)  # label='Fit'
                ax2.set_xscale('log')
                ax2.set_yscale('log')
                if j == 1:
                    ax2.set_ylim(20, 25000)
                elif j == 2:
                    ax2.set_ylim(150, 250000)
                ax2.axes.yaxis.set_visible(False)
                ax.text(0.8, 0.2, correlation_text, transform=ax.transAxes, color="red", fontsize=40)
            elif kkkk == 3:
                ax3 = ax2.twinx()
                ax3.tick_params(direction='in', which="major", length=15, width=3, pad=8)
                # 以下是两个轴都进行变换的线性拟合代码
                ax3.loglog(np.unique(x), np.exp(np.polyval(z1, np.log(np.unique(x)))), 'mediumblue', linewidth=5)  # label='Fit'
                ax3.set_xscale('log')
                ax3.set_yscale('log')
                if j == 1:
                    ax3.set_ylim(20, 25000)
                elif j == 2:
                    ax3.set_ylim(150, 250000)
                ax3.axes.yaxis.set_visible(False)
                ax.text(0.8, 0.1, correlation_text, transform=ax.transAxes, color="mediumblue", fontsize=40)
            kkkk += 1
    ax.set_xlabel('FlRate (fl' + r'$\cdot$' + r'min$^{-1}$)', fontsize=40)
    ax.set_xlim(0.5, 300)
    plt.yticks(fontsize=30)
    plt.xticks(fontsize=30)
    color_bar = ["black", "g", "red", "blue"]
    if j == 0:
        ax.set_ylabel(f"{name[j + 1]} (km)", fontsize=40)
    elif j == 1:
        ax.set_ylabel(f"{name[j + 1]} (km$^{2}$)", fontsize=40)
    elif j == 2:
        ax.set_ylabel(f"{name[j + 1]} (km$^{3}$)", fontsize=40)
    # plt.legend() # 指定legend的位置,读者可以自己help它的用法
    # 下面有了那个Area20和FlRate了就不要再在title上写了
    # ax.set_title(f"FlRate & {name[j + 1]}", fontsize=40)
    # Maturity stage
    # Development stage
    # Dissipation stage
    if j == 0:
        # ax.text(0.8, 0.1, f"b = {round(intercept, 2)}", transform=ax.transAxes, fontsize=40)
        # ax.text(0.8, 0.2, f"k = {round(slope, 2)}", transform=ax.transAxes, fontsize=40)
        # ax.text(0.8, 0.3, correlation_text, transform=ax.transAxes, fontsize=40)
        pass
    elif j == 1:
        # ax.text(0.8, 0.1, correlation_text, transform=ax.transAxes, fontsize=40)
        # ax.text(0.8, 0.2, f"b = {round(intercept, 2)}", transform=ax.transAxes, fontsize=40)
        # ax.text(0.8, 0.3, f"k = {round(slope, 2)}", transform=ax.transAxes, fontsize=40)
        ax.set_ylim(20, 25000)
    elif j == 2:
        # ax.text(0.8, 0.1, correlation_text, transform=ax.transAxes, fontsize=40)
        # ax.text(0.8, 0.2, f"b = {round(intercept, 2)}", transform=ax.transAxes, fontsize=40)
        # ax.text(0.8, 0.3, f"k = {round(slope, 2)}", transform=ax.transAxes, fontsize=40)
        # 这里进行了y的范围的设定
        ax.set_ylim(150, 250000)
    if j == 0:
        for_j_and_mid(name_data, color_bar, ax, 2.5)
        ax.set_ylim(0, 21)
        ax.text(0.04, 0.9, abcd[0], transform=ax.transAxes, fontsize=50)
        # levels = [50, 500, 1500, 3000]
        # _contour_(ax, x, y, 220, 20, 2, 1, levels)
    elif j == 1:
        for_j_and_mid(name_data, color_bar, ax, 2.5)
        ax.text(0.04, 0.9, abcd[1], transform=ax.transAxes, fontsize=50)
        # levels = [50, 500, 1500, 3000]
        # _contour_(ax, x, y, 220, 8000, 2, 200, levels)
    elif j == 2:
        for_j_and_mid(name_data, color_bar, ax, 2.5)
        ax.text(0.04, 0.9, abcd[2], transform=ax.transAxes, fontsize=50)
        # levels = [50, 500, 1500, 3000]
        # _contour_(ax, x, y, 220, 100000, 2, 5000, levels)
    # plt.show()
    coefficient += 1
    # os.system("pause")
plt.tight_layout()
plt.savefig(f"C:\\Users\\lvyih\\Desktop\\{name[0]}_{name[1]}_{name[2]}_all_stage.png", bbox_inches="tight",
            dpi=50)

"""ax.text(
        0.2, 0.1, 'some text',
        horizontalalignment='center',  # 水平居中
        verticalalignment='center',  # 垂直居中
        transform=ax.transAxes  # 使用相对坐标
    )"""
