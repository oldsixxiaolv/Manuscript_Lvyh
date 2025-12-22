# -- coding:utf-8 --
from pyhdf.SD import SD, SDC
import numpy as np
import math



def read_shuju1():
    data = SD('/root/git/Project_develop/manuscript_review/Excel_and_data_tropical_newest_no_executable/TRMM_tropical_convection_dataset.hdf', SDC.READ)
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
    index6 = list(np.where(npixels_40 == 0))
    index7 = list(np.where(maxht40 == 0))
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
    index_add1 = list(np.where(r > 0.56))
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
    return r, maxht20, maxht30, maxht40, npixels_20, npixels_30, npixels_40, flash_20, flash_30, flash_40, minir,\
        flashrate, maxht20_minux_maxht40, ellip_20, ellip_30, ellip_40, maxdbz,\
        maxht, npx40_divide_npx20, npx40_divide_npx30, ellip_20_maxht20, ellip_30_maxht30, ellip_40_maxht40, Volume20, Volume40
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
    file_path = r"/root/git/Project_develop/manuscript_review/Excel_and_data_tropical_newest_no_executable/duqu_last.xlsx"
    df = pd.read_excel(file_path, usecols=[1], names=None)
    dali = df.values.tolist()
    result = []
    for i in dali:
        result.append(i[0])
    return result


def ten_nine(x):
    to_sort_data = sorted(x)
    return to_sort_data[5165], to_sort_data[98158]

