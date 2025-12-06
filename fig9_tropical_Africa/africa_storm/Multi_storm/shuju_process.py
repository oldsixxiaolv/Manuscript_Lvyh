# -- coding:utf-8 --
from pyhdf.SD import SD, SDC
import numpy as np
import math


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
    indexx = duqu_excel()
    # 把hdf文件中的闪电频数数据提取出来
    flashcount = data.select('flashcount')[:]
    """这里我们可以看到对于一个可迭代的变量索引可以使用一个列表然后就可以直接赋值。"""
    flashcount = flashcount[index]
    flashcount = flashcount[indexx]
    # 把hdf文件中有雷暴的经度的数据弄出来
    latitude = data.select("latitude")[:]
    latitude = latitude[index]
    latitude = latitude[indexx]
    # 把hdf文件中有雷暴的纬度的数据弄出来
    longitude = data.select("longitude")[:]
    longitude = longitude[index]
    longitude = longitude[indexx]
    # 把hdf文件中总的降水数据提取出来
    volrain = data.select('volrain')[:]
    volrain = volrain[index]
    volrain = volrain[indexx]
    # 把hdf文件中的层云降水数据提取出来
    rainstrat = data.select('rainstrat')[:]
    rainstrat = rainstrat[index]
    rainstrat = rainstrat[indexx]
    # 把hdf文件中的对流降水数据分离出来
    rainconv = data.select('rainconv')[:]
    rainconv = rainconv[index]
    rainconv = rainconv[indexx]
    # 把hdf文件中的观测时间数据提取出来
    viewtime = data.select('viewtime')[:]
    viewtime = viewtime[index]
    viewtime = viewtime[indexx]
    # 把hdf文件是否升轨数据提取出来
    boost = data.select("boost")[:]
    boost = boost[index]
    boost = boost[indexx]
    # 把hdf文件的最大顶高数据提取出来（这个数据老师说第一个maxht不用操作）
    maxht = data.select("maxht")[:]
    maxht = maxht[index]
    maxht = maxht[indexx]
    maxht20 = data.select("maxht20")[:]
    maxht20 = maxht20[index]
    maxht20 = maxht20[indexx]
    maxht30 = data.select("maxht30")[:]
    maxht30 = maxht30[index]
    maxht30 = maxht30[indexx]
    maxht40 = data.select("maxht40")[:]
    maxht40 = maxht40[index]
    maxht40 = maxht40[indexx]
    # 这个是云的顶层的亮温，可以通过知晓这个温度的高低来判断云顶的高度，温度越低意味着云越高
    minir = data.select("minir")[:]
    minir = minir[index]
    minir = minir[indexx]
    # 这个先不用
    npixels_40 = data.select("npixels_40")[:]
    npixels_40 = npixels_40[index]
    npixels_40 = npixels_40[indexx]
    npixels_40 = npixels_size(npixels_40, boost)
    npixels_40_K = np.multiply(npixels_40, 4)
    npixels_40_K = np.divide(npixels_40_K, math.pi)
    npixels_40_R = np.sqrt(npixels_40_K)
    npixels_20 = data.select("npixels_20")[:]
    npixels_20 = npixels_20[index]
    npixels_20 = npixels_20[indexx]
    npixels_20 = npixels_size(npixels_20, boost)
    npixels_20_K = np.multiply(npixels_20, 4)
    npixels_20_K = np.divide(npixels_20_K, math.pi)
    # 这里的npixels_20_R其实就是我们的水平尺度（按照面积想象成圆形然后换算成直径的）
    npixels_20_R = np.sqrt(npixels_20_K)
    npixels_30 = data.select("npixels_30")[:]
    npixels_30 = npixels_30[index]
    npixels_30 = npixels_30[indexx]
    npixels_30 = npixels_size(npixels_30, boost)
    npixels_30_K = np.multiply(npixels_30, 4)
    npixels_30_K = np.divide(npixels_30_K, math.pi)
    npixels_30_R = np.sqrt(npixels_30_K)
    # 20,30,40格点数
    # n20d = data.select("n20dbz")[:]
    # n30d = data.select("n30dbz")[:]
    # n40d = data.select("n40dbz")[:]
    # n20db = []
    # n30db = []
    # n40db = []
    # max_n(n20d, n20db)
    # max_n(n30d, n30db)
    # max_n(n40d, n40db)
    # n20dbz = np.array(n20db)
    # n30dbz = np.array(n30db)
    # n40dbz = np.array(n40db)
    # 雷暴云最大雷达回波值
    # maxd = data.select("maxdbz")[:]
    # maxdb  = []
    # for k in maxd:
    # maxdb.append(max(k))
    # maxdbz = np.array(maxdb)
    # 低层有降水的面积
    maxnsrain = data.select("maxnsrain")[:]
    maxnsrain = maxnsrain[index]
    maxnsrain = maxnsrain[indexx]
    # 将所有的rainconv数据每一项除以volrain变成新的数据r数组
    r = np.divide(rainconv, volrain)
    maxdbz20_minus_maxdbz40 = np.subtract(maxht20, maxht40)
    maxdbz20_minus_maxdbz40_divide_maxdbz40 = np.divide(maxdbz20_minus_maxdbz40, maxht40)
    maxht_divide_maxdbz20_minus_maxdbz40_divide_maxdbz40 = np.divide(maxht, maxdbz20_minus_maxdbz40_divide_maxdbz40)
    ellipsoidal_ratio_20 = np.divide(maxdbz20_minus_maxdbz40, npixels_20_R)
    ellipsoidal_ratio_30 = np.divide(maxdbz20_minus_maxdbz40, npixels_30_R)
    ellipsoidal_ratio_40 = np.divide(maxdbz20_minus_maxdbz40, npixels_40_R,
                                     out=np.zeros_like(npixels_40_R), where=npixels_40_R != 0)
    # 闪电频数=闪电数/viewtime*60，我们使用每分钟的闪电频数
    flashfrequence = np.divide(flashcount * 60, viewtime)
    return latitude, longitude


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
    return pixel


def duqu_excel():
    import pandas as pd
    file_path = r'E:\\Programing\\school_creation_python311\\' \
                'xiaochuang\\2024_thesis\\Excel_and_data_tropical_newest_no_executable\\' \
                'duqu_last.xlsx'
    df = pd.read_excel(file_path, usecols=[1], names=None)
    dali = df.values.tolist()
    result = []
    result_result = []
    for i in dali:
        result.append(i[0])
    for q in range(0, 135089):
        if q not in result:
            result_result.append(q)
    return result_result
