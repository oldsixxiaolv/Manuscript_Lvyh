from pyhdf.SD import SD, SDC
import numpy as np


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


def read_shuju():
    filepath = r'/root/git/Project_develop/TRMM_tropical_convection_dataset.hdf'
    data = SD(filepath, SDC.READ)
    longitude = data.select("longitude")[:]
    latitude = data.select("latitude")[:]
    landocean = list(data.select("landocean")[:])
    landocean = np.array(list(map(int, landocean)))
    flashcount = data.select('flashcount')[:]
    flashcount = np.array(list(map(int, flashcount)))
    maxht20 = data.select("maxht20")[:]
    volrain = data.select('volrain')[:]
    rainconv = data.select('rainconv')[:]
    npixels_20 = data.select("npixels_20")[:]
    npixels_40 = data.select("npixels_40")[:]
    viewtime = data.select("viewtime")[:]
    maxht40 = data.select("maxht40")[:]
    # boost = data.select("boost")[:]
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
    # index8 = list(np.where(flashcount != 0))
    # index8 = list(np.where(npixels_20 > 0))
    # index9 = list(np.where(r > 0))
    # index10 = list(np.where(r < 1))
    index11 = list(np.where(maxht20 != 0))
    index12 = list(np.where(npixels_20 != 0))
    index13 = list(np.where(~np.isnan(viewtime)))
    # 两个数组取交集
    index = list(set(index1[0]) & set(index2[0]) & set(index3[0]) & set(index4[0]) &
                 set(index5[0]) & set(index6[0]) & set(index7[0]) &
                 set(index11[0]) & set(index12[0]) & set(index13[0]))
    index = sorted(index)
    # indexx = duqu_excel()
    volrain = data.select('volrain')[:]
    volrain = volrain[index]
    # volrain = volrain[indexx]
    # 把hdf文件中的层云降水数据提取出来
    rainstrat = data.select('rainstrat')[:]
    rainstrat = rainstrat[index]
    # rainstrat = rainstrat[indexx]
    # 把hdf文件中的对流降水数据分离出来
    rainconv = data.select('rainconv')[:]
    rainconv = rainconv[index]
    # rainconv = rainconv[indexx]
    # 将所有的rainconv数据每一项除以volrain变成新的数据r数组
    r = np.divide(rainconv, volrain)
    # index_r_first = list(np.where(r > 0))
    # index_r_second = list(np.where(r < 1))
    # index_r_all = list(set(index_r_first[0]) & set(index_r_second[0]))
    # r = r[index_r_all]
    return r


def duqu_excel():
    """此用来读取我们聚类好的索引以便在read_shuju()函数中进行筛选"""
    import pandas as pd
    file_path = r"/root/git/Project_develop/manuscript_review/Excel_and_data_tropical_newest_no_executable/duqu_last.xlsx"
    df = pd.read_excel(file_path, usecols=[1], names=None)
    dali = df.values.tolist()
    result = []
    for i in dali:
        result.append(i[0])
    return result
