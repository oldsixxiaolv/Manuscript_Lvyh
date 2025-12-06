# -- coding:utf-8 --
from pyhdf.SD import SD, SDC
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
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
    file_path = r"C:\Users\lvyih\Desktop\duqu_last.xlsx"
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
    data = SD('C:\\Users\\lvyih\\Desktop\\TRMM_tropical_convection_dataset.hdf', SDC.READ)
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
    index2 = list(np.where(latitude > -30))
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
    inde = list(np.where(r > 0.66))
    index_add2 = list(np.where(minir > 0))
    inded = list(set(inde[0]) & set(index_add2[0]))
    flashrate = flashrate[inded]
    maxht40 = maxht40[inded]
    maxht20 = maxht20[inded]
    maxht30 = maxht30[inded]
    npixels_20 = npixels_20[inded]
    npixels_30 = npixels_30[inded]
    npixels_40 = npixels_40[inded]
    npixels_20_R = npixels_20_R[inded]
    npixels_30_R = npixels_30_R[inded]
    npixels_40_R = npixels_40_R[inded]
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
    return flashrate, n20_volume, npixels_30, n40_volume


def for_(data, stage):
    mid = []
    for i in data:
        mid.append(i[stage])
    return mid


def Ex(x, a, b, c):
    return a * np.exp(b * x) + c


"""程序开始"""
# 程序发起点
name = ["Flash", "Vpx20", "Spx30", "Vpx40"]
data = read_shuju()
# data_add = read_shuju_ADD()
# data_mid = []
# for i, j in zip(data, data_add):
#     data_mid.append(np.concatenate((i, j)))
# data = data_mid
stage1 = duqu_excel_julei(r"C:\Users\lvyih\Desktop\stage1.xlsx")
stage2 = duqu_excel_julei(r"C:\Users\lvyih\Desktop\stage2.xlsx")
stage3 = duqu_excel_julei(r"C:\Users\lvyih\Desktop\stage3.xlsx")
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
stage = ["All Stage", "Pre-Mature Stage", "Maturity stage", "Post-Mature Stage"]
coefficient = 0
for j in range(0, 3):
    mid = 1
    for i in name_data:
        fig = plt.figure(coefficient, figsize=(30, 30))
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 21
        plt.rc('axes', linewidth=3)
        flashrate = i[0]
        parameter = i[j+1]
        # 用4次多项式拟合
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
        ax.semilogx(np.unique(x), np.polyval(z1, np.unique(log_x)), 'r')  # label='Fit'
        ax.set_xscale('log')
        ax.set_xlabel("Flashrate/fl min$^{-1}$", fontsize=40)
        plt.yticks(fontsize=30)
        plt.xticks(fontsize=30)
        color_bar = ["black", "g", "red", "blue"]
        # if j <= 2:
        #     ax.set_ylabel(f"{name[j+1]}/km", fontsize=40)
        if j == 2:
            # ax.set_ylabel(f"{name[j+1]}/km" + "$^{2}$", fontsize=40)
            ax.set_ylabel(f"{name[j + 1]}/km" + "$^{3}$", fontsize=40)
        elif j == 4:
            ax.set_ylabel(f"{name[j + 1]}/K", fontsize=40)
        # plt.legend() # 指定legend的位置,读者可以自己help它的用法
        ax.set_title(f"Flashrate & {name[j+1]} {stage[mid-1]}", fontsize=40)
        # Maturity stage
        # Development stage
        # Dissipation stage
        if j == 0:
            ax.text(0.1, 0.9, equation_text, transform=ax.transAxes, fontsize=35)
            ax.text(0.1, 0.8, correlation_text, transform=ax.transAxes, fontsize=35)
        elif j == 1:
            ax.text(0.1, 0.9, equation_text, transform=ax.transAxes, fontsize=35)
            ax.text(0.1, 0.8, correlation_text, transform=ax.transAxes, fontsize=35)
        elif j == 2:
            ax.text(0.1, 0.9, equation_text, transform=ax.transAxes, fontsize=35)
            ax.text(0.1, 0.8, correlation_text, transform=ax.transAxes, fontsize=35)
        elif j == 3:
            ax.text(0.1, 0.9, equation_text, transform=ax.transAxes, fontsize=35)
            ax.text(0.1, 0.8, correlation_text, transform=ax.transAxes, fontsize=35)
        else:
            ax.text(0.1, 0.9, equation_text, transform=ax.transAxes, fontsize=35)
            ax.text(0.1, 0.8, correlation_text, transform=ax.transAxes, fontsize=35)
        if j == 3 and mid == 1:
            for kkk, www in zip(name_data[3:0:-1], color_bar[3:0:-1]):
                flashrate = kkk[0]
                parameter = kkk[j + 1]
                # 用4次多项式拟合
                x = flashrate
                log_x = np.log(x)
                y = parameter
                ax.scatter(x, y, color=www, s=2)  # label='hetu_original_boxplot values'
            ax.set_ylim(30, 65)
            # ax.legend(loc="upper left", bbox_to_anchor=(0.45, 0.3), prop={"family": "Times New Roman", "size": 40})
        elif j == 3 and mid == 2:
            ax.scatter(x, y, color=color_bar[mid - 1], s=4)  # label='hetu_original_boxplot values'
            ax.set_ylim(30, 65)
            # ax.legend(loc="upper left", bbox_to_anchor=(0.45, 0.3), prop={"family": "Times New Roman", "size": 40})
        elif j == 3 and mid == 3:
            ax.scatter(x, y, color=color_bar[mid - 1], s=4)  # label='hetu_original_boxplot values'
            ax.set_ylim(30, 65)
            # ax.legend(loc="upper left", bbox_to_anchor=(0.45, 0.3), prop={"family": "Times New Roman", "size": 40})
        elif j == 3 and mid == 4:
            ax.scatter(x, y, color=color_bar[mid - 1], s=4)  # label='hetu_original_boxplot values'
            ax.set_ylim(30, 65)
            # ax.legend(loc="upper left", bbox_to_anchor=(0.45, 0.3), prop={"family": "Times New Roman", "size": 40})
        elif j == 3 and mid == 5:
            ax.scatter(x, y, color=color_bar[mid - 1], s=4)  # label='hetu_original_boxplot values'
            ax.set_ylim(30, 65)
            # ax.legend(loc="upper left", bbox_to_anchor=(0.45, 0.3), prop={"family": "Times New Roman", "size": 40})
        elif j == 4 and mid == 1:
            for kkk, www in zip(name_data[3:0:-1], color_bar[3:0:-1]):
                flashrate = kkk[0]
                parameter = kkk[j + 1]
                # 用4次多项式拟合
                x = flashrate
                log_x = np.log(x)
                y = parameter
                ax.scatter(x, y, color=www, s=2)  # label='hetu_original_boxplot values'
            # ax.legend(loc="upper left", bbox_to_anchor=(0.4, 1), prop={"family": "Times New Roman", "size": 40})
            ax.set_ylim(160, 280)
        elif j == 4 and mid == 2:
            ax.scatter(x, y, color=color_bar[mid - 1], s=4)  # label='hetu_original_boxplot values'
            # ax.legend(loc="upper left", bbox_to_anchor=(0.4, 1), prop={"family": "Times New Roman", "size": 40})
            ax.set_ylim(160, 280)
        elif j == 4 and mid == 3:
            ax.scatter(x, y, color=color_bar[mid - 1], s=4)  # label='hetu_original_boxplot values'
            # ax.legend(loc="upper left", bbox_to_anchor=(0.4, 1), prop={"family": "Times New Roman", "size": 40})
            ax.set_ylim(160, 280)
        elif j == 4 and mid == 4:
            ax.scatter(x, y, color=color_bar[mid - 1], s=4)  # label='hetu_original_boxplot values'
            # ax.legend(loc="upper left", bbox_to_anchor=(0.4, 1), prop={"family": "Times New Roman", "size": 40})
            ax.set_ylim(160, 280)
        elif j == 4 and mid == 5:
            ax.scatter(x, y, color=color_bar[mid - 1], s=2)
            ax.set_ylim(170, 260)
        elif j == 5 and mid == 1:
            for kkk, www in zip(name_data, color_bar):
                flashrate = kkk[0]
                parameter = kkk[j + 1]
                # 用4次多项式拟合
                x = flashrate
                log_x = np.log10(x)
                y = parameter
                ax.scatter(x, y, color=www, s=2)
            ax.set_ylim(0, 30)
        elif j == 5 and mid == 2:
            ax.scatter(x, y, color=color_bar[mid-1], s=2)
            ax.set_ylim(0, 30)
        elif j == 5 and mid == 3:
            ax.scatter(x, y, color=color_bar[mid-1], s=2)
            ax.set_ylim(0, 30)
        elif j == 5 and mid == 4:
            ax.scatter(x, y, color=color_bar[mid-1], s=2)
            ax.set_ylim(0, 30)
        elif j == 5 and mid == 5:
            ax.scatter(x, y, color=color_bar[mid - 1], s=2)
            ax.set_ylim(0, 30)
        elif j == 0 and mid == 1:
            for kkk, www in zip(name_data[3:0:-1], color_bar[3:0:-1]):
                flashrate = kkk[0]
                parameter = kkk[j + 1]
                # 用4次多项式拟合
                x = flashrate
                log_x = np.log(x)
                y = parameter
                ax.scatter(x, y, color=www, s=2)  # label='hetu_original_boxplot values'
            ax.set_ylim(0, 100000)
            # ax.legend(loc="upper left", bbox_to_anchor=(0.45, 0.3), prop={"family": "Times New Roman", "size": 40})
        elif j == 0 and mid == 2:
            ax.scatter(x, y, color=color_bar[mid-1], s=4)  # label='hetu_original_boxplot values'
            ax.set_ylim(0, 100000)
            # ax.legend(loc="upper left", bbox_to_anchor=(0.45, 0.3), prop={"family": "Times New Roman", "size": 40})
        elif j == 0 and mid == 3:
            ax.scatter(x, y, color=color_bar[mid-1], s=4)  # label='hetu_original_boxplot values'
            ax.set_ylim(0, 100000)
            # ax.legend(loc="upper left", bbox_to_anchor=(0.45, 0.3), prop={"family": "Times New Roman", "size": 40})
        elif j == 0 and mid == 4:
            ax.scatter(x, y, color=color_bar[mid-1], s=4)  # label='hetu_original_boxplot values'
            ax.set_ylim(0, 100000)
            # ax.legend(loc="upper left", bbox_to_anchor=(0.45, 0.3), prop={"family": "Times New Roman", "size": 40})
        elif j == 0 and mid == 5:
            ax.scatter(x, y, color=color_bar[mid-1], s=4)  # label='hetu_original_boxplot values'
            ax.set_ylim(0, 21)
            # ax.legend(loc="upper left", bbox_to_anchor=(0.45, 0.3), prop={"family": "Times New Roman", "size": 40})
        elif j == 1 and mid == 1:
            for kkk, www in zip(name_data[3:0:-1], color_bar[3:0:-1]):
                flashrate = kkk[0]
                parameter = kkk[j + 1]
                # 用4次多项式拟合
                x = flashrate
                log_x = np.log(x)
                y = parameter
                ax.scatter(x, y, color=www, s=4)
            ax.set_ylim(0, 4000)
            # ax.legend(loc="upper left", bbox_to_anchor=(0.45, 0.3), prop={"family": "Times New Roman", "size": 40})
        elif j == 1 and mid == 2:
            ax.scatter(x, y, color=color_bar[mid-1], s=4)
            ax.set_ylim(0, 4000)
            # ax.legend(loc="upper left", bbox_to_anchor=(0.45, 0.3), prop={"family": "Times New Roman", "size": 40})
        elif j == 1 and mid == 3:
            ax.scatter(x, y, color=color_bar[mid-1], s=4)
            ax.set_ylim(0, 4000)
            # ax.legend(loc="upper left", bbox_to_anchor=(0.45, 0.3), prop={"family": "Times New Roman", "size": 40})
        elif j == 1 and mid == 4:
            ax.scatter(x, y, color=color_bar[mid-1], s=4)
            ax.set_ylim(0, 4000)
            # ax.legend(loc="upper left", bbox_to_anchor=(0.45, 0.3), prop={"family": "Times New Roman", "size": 40})
        elif j == 1 and mid == 5:
            ax.scatter(x, y, color=color_bar[mid-1], s=4)  # label='hetu_original_boxplot values'
            ax.set_ylim(0, 21)
            # ax.legend(loc="upper left", bbox_to_anchor=(0.45, 0.3), prop={"family": "Times New Roman", "size": 40})
        elif j == 2 and mid == 1:
            for kkk, www in zip(name_data[3:0:-1], color_bar[3:0:-1]):
                flashrate = kkk[0]
                parameter = kkk[j + 1]
                # 用4次多项式拟合
                x = flashrate
                log_x = np.log(x)
                y = parameter
                ax.scatter(x, y, color=www, s=4)
            ax.set_ylim(0, 10000)
            # ax.legend(loc="upper left", bbox_to_anchor=(0, 1), prop={"family": "Times New Roman", "size": 40})
        elif j == 2 and mid == 2:
            ax.scatter(x, y, color=color_bar[mid-1], s=4)
            ax.set_ylim(0, 10000)
            # ax.legend(loc="upper left", bbox_to_anchor=(0, 1), prop={"family": "Times New Roman", "size": 40})
        elif j == 2 and mid == 3:
            ax.scatter(x, y, color=color_bar[mid-1], s=4)
            ax.set_ylim(0, 10000)
            # ax.legend(loc="upper left", bbox_to_anchor=(0, 1), prop={"family": "Times New Roman", "size": 40})
        elif j == 2 and mid == 4:
            ax.scatter(x, y, color=color_bar[mid-1], s=4)
            ax.set_ylim(0, 10000)
            # ax.legend(loc="upper left", bbox_to_anchor=(0, 1), prop={"family": "Times New Roman", "size": 40})
        elif j == 2 and mid == 5:
            ax.scatter(x, y, color=color_bar[mid-1], s=4)  # label='hetu_original_boxplot values'
            ax.set_ylim(0, 21)
            # ax.legend(loc="upper left", bbox_to_anchor=(0.45, 1), prop={"family": "Times New Roman", "size": 40})
        mid += 1
    plt.tight_layout()
    plt.savefig(f"C:\\Users\\lvyih\\Desktop\\{name[j+1]}_all_stage.png", bbox_inches="tight",
                dpi=400)
    # plt.show()
    coefficient += 1
    # os.system("pause")
"""ax.text(
        0.2, 0.1, 'some text',
        horizontalalignment='center',  # 水平居中
        verticalalignment='center',  # 垂直居中
        transform=ax.transAxes  # 使用相对坐标
    )"""
