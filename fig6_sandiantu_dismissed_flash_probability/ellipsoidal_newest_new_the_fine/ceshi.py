# -- coding:utf-8 --
from pyhdf.SD import SD, SDC
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def duqu_excel():
    import pandas as pd
    file_path = r"C:\Users\lvyih\Desktop\duqu.xlsx"
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
    index1 = np.where(longitude > -30)
    index2 = np.where(latitude > -40)
    index3 = np.where(landocean == 1)
    index4 = np.where(longitude < 60)
    index5 = np.where(latitude < 38)
    # 把没有闪电数据的但是有maxht40的和没有maxht40但有闪电数据的去除
    index6 = list(np.where(flashcount != 0))
    index7 = list(np.where(maxht40 != 0))
    # index8 = list(np.where(r < 0.98))
    # index9 = list(np.where(r >= 0.8))
    indexw = duqu_excel()
    # 两个数组取交集
    index = list(set(index1[0]) & set(index2[0]) & set(index4[0]) & set(index5[0]) & set(index6[0])
                        & set(index7[0]))
    #  & set(index8[0]) & set(index9[0])
    index = sorted(index)
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
    npixels_20 = data.select("npixels_20")[:]
    npixels_20 = npixels_20[index]
    npixels_20 = npixels_20[indexw]
    npixels_20 = npixels_size(npixels_20, boost)
    npixels_30 = data.select("npixels_30")[:]
    npixels_30 = npixels_30[index]
    npixels_30 = npixels_30[indexw]
    npixels_30 = npixels_size(npixels_30, boost)
    maxnsrain = data.select("maxnsrain")[:]
    maxnsrain = maxnsrain[index]
    maxnsrain = maxnsrain[indexw]
    flashcount = flashcount.astype(float)
    flashrate = np.divide(flashcount, viewtime)
    flashrate = np.multiply(flashrate, 60)
    inde = list(np.where(r > former))
    ind = list(np.where(r <= latter))
    inded = list(set(ind[0]) & set(inde[0]))
    inded = sorted(inded)
    flashrate = flashrate[inded]
    maxht40 = maxht40[inded]
    maxht20 = maxht20[inded]
    maxht30 = maxht30[inded]
    npixels_20 = npixels_20[inded]
    npixels_30 = npixels_30[inded]
    npixels_40 = npixels_40[inded]
    return flashrate, maxht20, maxht30, maxht40, npixels_20, npixels_30, npixels_40


# fig = plt.figure(1, figsize=(5, 5))
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
data_test = read_shuju(0.9, 1)
flashrate = data_test[0]
maxht20 = data_test[1]
maxht30 = data_test[2]
maxht40 = data_test[3]
npixels_20 = data_test[4]
npixels_30 = data_test[5]
npixels_40 = data_test[6]
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1)
plt.tick_params(labelsize=12)
scatter_xy = np.vstack([maxht30, maxht20])
scatter_z = gaussian_kde(scatter_xy)(scatter_xy)
scatter = ax.scatter(maxht30, maxht20, marker='o', s=15, c=scatter_z, cmap='Spectral_r')
ax.set_xlim(0, None)
ax.set_ylim(0, None)
plt.xlabel('maxht30/km', fontsize=12)
plt.ylabel('maxht20/km', fontsize=12)
plt.savefig(r"C:\\Users\\lvyih\\Desktop\\all_stage_ceshi.png", dpi=400)
