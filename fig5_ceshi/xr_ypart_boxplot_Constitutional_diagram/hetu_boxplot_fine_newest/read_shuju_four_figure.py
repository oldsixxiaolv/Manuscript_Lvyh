# -- coding:utf-8 --
from pyhdf.SD import SD, SDC
import numpy as np
import math


def read_shuju():
    filepath = r'/root/git/Project_develop/manuscript_review/Excel_and_data_tropical_newest_no_executable/TRMM_tropical_convection_dataset.hdf'
    data = SD(filepath, SDC.READ)
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
    # indexx = duqu_excel()
    # 把hdf文件中的闪电频数数据提取出来
    flashcount = data.select('flashcount')[:]
    """这里我们可以看到对于一个可迭代的变量索引可以使用一个列表然后就可以直接赋值。"""
    flashcount = flashcount[index]
    # flashcount = flashcount[indexx]
    # 把hdf文件中总的降水数据提取出来
    # volrain = data.select('volrain')[:]
    volrain = volrain[index]
    # volrain = volrain[indexx]
    # 把hdf文件中的对流降水数据分离出来
    # rainconv = data.select('rainconv')[:]
    rainconv = rainconv[index]
    # rainconv = rainconv[indexx]
    # 把hdf文件中的观测时间数据提取出来
    # viewtime = data.select('viewtime')[:]
    viewtime = viewtime[index]
    # viewtime = viewtime[indexx]
    # 把hdf文件是否升轨数据提取出来
    boost = data.select("boost")[:]
    boost = boost[index]
    # boost = boost[indexx]
    maxht = data.select("maxht")[:]
    maxht = maxht[index]
    # maxht = maxht[indexx]
    # maxht20 = data.select("maxht20")[:]
    maxht20 = maxht20[index]
    # maxht20 = maxht20[indexx]
    # maxht40 = data.select("maxht40")[:]
    maxht40 = maxht40[index]
    # maxht40 = maxht40[indexx]
    # npixels_40 = data.select("npixels_40")[:]
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
    # npixels_20 = data.select("npixels_20")[:]
    npixels_20 = npixels_20[index]
    # npixels_20 = npixels_20[indexx]
    npixels_20 = npixels_size(npixels_20, boost)
    npixels_20_K = np.multiply(npixels_20, 4)
    npixels_20_K = np.divide(npixels_20_K, math.pi)
    npixels_20_R = np.sqrt(npixels_20_K)
    maxdbz = data.select("maxdbz")[:]
    maxdbz = maxdbz[index]
    # maxdbz = maxdbz[indexx]
    n40 = data.select("n40dbz")[:]
    n40 = n40[index]
    # n40 = n40[indexx]
    n40_mid = []
    for i, j in zip(n40, boost):
        if j == 1:
            mid = sum(i * 5 * 5 * 1.25)
        else:
            mid = sum(i * 4.3 * 4.3 * 1.25)
        n40_mid.append(mid)
    n40_volume_divide_1000 = np.array(n40_mid) / 1000
    n20 = data.select("n20dbz")[:]
    n20 = n20[index]
    # n20 = n20[indexx]
    n20_mid = []
    for u, v in zip(n20, boost):
        if v == 1:
            mid_mid = sum(u * 5 * 5 * 1.25)
        else:
            mid_mid = sum(u * 4.3 * 4.3 * 1.25)
        n20_mid.append(mid_mid)
    n20_volume_divide_10000 = np.array(n20_mid) / 10000
    minir = data.select("minir")[:]
    minir = minir[index]
    # minir = minir[indexx]
    # 将所有的rainconv数据每一项除以volrain变成新的数据r数组
    r = np.divide(rainconv, volrain)
    # 闪电频数=闪电数/viewtime*60，我们使用每分钟的闪电频数
    flashfrequence = np.divide(flashcount * 60, viewtime)
    flashfrequence_20 = np.divide(flashfrequence, npixels_20)
    flashfrequence_20 = np.multiply(flashfrequence_20, 100)
    flashfrequence_40 = np.divide(flashfrequence, npixels_40)
    flashfrequence_40 = np.multiply(flashfrequence_40, 100)
    flashfrequence_30 = np.divide(flashfrequence, npixels_30)
    flashfrequence_30 = np.multiply(flashfrequence_30, 100)
    return r, maxht20, npixels_20 / 1000, \
        maxht40, npixels_40 / 100, \
        n20_volume_divide_10000, n40_volume_divide_1000, \
        flashfrequence, flashfrequence_40
    # n20dbz, n30dbz, n40dbz, maxdbz


def max_n(shuju1, shuju2):
    k = 0
    for i in shuju1:
        if k == 0:
            shuju2.append(0)
            k += 1
        else:
            shuju2.append(max(i))


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


def duqu_excel():
    import pandas as pd
    file_path = r'/root/git/Project_develop/manuscript_review/Excel_and_data_tropical_newest_no_executable/duqu_last.xlsx'
    df = pd.read_excel(file_path, usecols=[1], names=None)
    dali = df.values.tolist()
    result = []
    for i in dali:
        result.append(i[0])
    return result


def ten_nine(x):
    to_sort_data = sorted(x)
    return to_sort_data[round(len(to_sort_data) * 0.05)], to_sort_data[round(len(to_sort_data) * 0.95)]
