# -- coding:utf-8 --
from pyhdf.SD import SD, SDC
import numpy as np
import matplotlib.pyplot as plt


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
    index2 = list(np.where(latitude > -30))
    index3 = list(np.where(landocean == 1))
    index4 = list(np.where(longitude < 50))
    index5 = list(np.where(latitude < 20))

    index1_Nanmei = np.where(longitude > -110)
    index2_Nanmei = list(np.where(latitude > -30))
    index4_Nanmei = list(np.where(longitude < -40))
    index5_Nanmei = list(np.where(latitude < 30))

    index1_Yindunixia = np.where(longitude > 65)
    index2_Yindunixia = list(np.where(latitude > -30))
    index4_Yindunixia = list(np.where(longitude < 150))
    index5_Yindunixia = list(np.where(latitude < 30))

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
    index_Africa = list(set(index1[0]) & set(index2[0]) & set(index3[0]) & set(index4[0]) &
                 set(index5[0]) & set(index6[0]) & set(index7[0]) & set(index8[0]) &
                 set(index11[0]) & set(index12[0]) & set(index13[0]))
    index_Nanmei = list(set(index1_Nanmei[0]) & set(index2_Nanmei[0]) & set(index3[0])
                        & set(index4_Nanmei[0]) & set(index5_Nanmei[0]) & set(index6[0])
                        & set(index7[0]) & set(index8[0]) & set(index11[0]) & set(index12[0])
                        & set(index13[0]))
    index_Yindunixia = list(set(index1_Yindunixia[0]) & set(index2_Yindunixia[0]) & set(index3[0])
                        & set(index4_Yindunixia[0]) & set(index5_Yindunixia[0]) & set(index6[0])
                        & set(index7[0]) & set(index8[0]) & set(index11[0]) & set(index12[0])
                        & set(index13[0]))
    baby = []
    for qqq in [index_Africa, index_Nanmei, index_Yindunixia]:
        index = sorted(qqq)
        r_mid = r[index]
        # 把hdf文件中的闪电频数数据提取出来
        flashcount = data.select('flashcount')[:]
        flashcount_mid = flashcount[index]
        # 把hdf文件中总的降水数据提取出来
        volrain = data.select('volrain')[:]
        volrain_mid = volrain[index]
        # 把hdf文件中的层云降水数据提取出来
        rainstrat = data.select('rainstrat')[:]
        rainstrat_mid = rainstrat[index]
        # 把hdf文件中的对流降水数据分离出
        rainconv = data.select('rainconv')[:]
        rainconv_mid = rainconv[index]
        # 把hdf文件中的观测时间数据提取出来
        viewtime = data.select('viewtime')[:]
        viewtime_mid = viewtime[index]
        # 把hdf文件是否升轨数据提取出来
        boost = data.select("boost")[:]
        boost_mid = boost[index]
        # 把hdf文件的最大顶高数据提取出来（这个数据老师说第一个maxht不用操作）
        maxht = data.select("maxht")[:]
        maxht_mid = maxht[index]
        maxht20 = data.select("maxht20")[:]
        maxht20_mid = maxht20[index]
        maxht30 = data .select("maxht30")[:]
        maxht30_mid = maxht30[index]
        maxht40 = data.select("maxht40")[:]
        maxht40_mid = maxht40[index]
        # 这个是云的顶层的亮温，可以通过知晓这个温度的高低来判断云顶的高度，温度越低意味着云越高
        minir = data.select("minir")[:]
        minir_mid = minir[index]
        # 这个先不用
        npixels_40 = data.select("npixels_40")[:]
        npixels_40 = npixels_40[index]
        npixels_40 = npixels_size(npixels_40, boost)
        npixels_20 = data.select("npixels_20")[:]
        npixels_20 = npixels_20[index]
        npixels_20 = npixels_size(npixels_20, boost)
        npixels_30 = data.select("npixels_30")[:]
        npixels_30 = npixels_30[index]
        npixels_30 = npixels_size(npixels_30, boost)
        maxnsrain = data.select("maxnsrain")[:]
        maxnsrain = maxnsrain[index]
        flashcount_mid = flashcount_mid.astype(float)
        flashrate = np.divide(60 * flashcount_mid, viewtime_mid)
        n20 = data.select("n20dbz")[:]
        n20 = n20[index]
        n40 = data.select("n40dbz")[:]
        n40 = n40[index]
        mdbz = data.select("mdbz")[:]
        mdbz = mdbz[index]
        inde = list(np.where(r_mid >= 0))
        index_add1 = list(np.where(minir_mid > 0))
        inded = list(set(inde[0]) & set(index_add1[0]))
        inded = sorted(inded)
        n20 = n20[inded]
        n40 = n40[inded]
        mdbz = mdbz[inded]
        r_mid = r_mid[inded]
        mid_list = []
        mid_list.append(n20)
        mid_list.append(n40)
        mid_list.append(mdbz)
        mid_list.append(r_mid)
        baby.append(mid_list)
    return baby


def top10(data_mid):
    output = []
    for w in range(0, 40):
        mid_list = []
        for i in data_mid:
            mid_list.append(i[w])
        mid_list = sorted(mid_list)[round(len(mid_list) * 0.9):]
        output.append(np.mean(np.array(mid_list)))
    return output


"""程序开始"""
# 程序发起点
plt.rc('axes', linewidth=3)
plt.tick_params(width=3)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 40
data = read_shuju()
# Africa
data1 = data[0]
# America
data2 = data[1]
# Maritime Continent
data3 = data[2]
name_data = [data1, data2, data3]
stage = ["Africa", "Americas", "Maritime_continent"]
Height = np.arange(0, 20, 0.5)
# 绘制overall average
mdbz_africa = np.mean(data1[2], axis=0)
mdbz_americas = np.mean(data2[2], axis=0)
mdbz_Maritime_continent = np.mean(data3[2], axis=0)
# 绘制top 10% average
# mdbz_africa_average = top10(data1[2])
# mdbz_americas_average = top10(data2[2])
# mdbz_Maritime_continent_average = top10(data3[2])
# np.set_printoptions(threshold=np.inf)
fig = plt.figure(figsize=(10, 15))
ax = fig.add_subplot(1, 1, 1)
# ax.plot(mdbz_all, Height, "black", marker="*")
# 绘制overall average
ax.plot(mdbz_africa, Height, "r", marker="^",
        label="Africa", linewidth=4.33, markersize=12)
ax.plot(mdbz_americas, Height, "b", marker="s",
        label="Americas", linewidth=4.33, markersize=12)
ax.plot(mdbz_Maritime_continent, Height, "g", marker="o",
        label="Maritime Continent", linewidth=4.33, markersize=12)
# 绘制top10% average
# ax.plot(mdbz_africa_average, Height, "r", marker="^",
#         linewidth=4.33, markersize=12)
# ax.plot(mdbz_americas_average, Height, "b", marker="s",
#         linewidth=4.33, markersize=12)
# ax.plot(mdbz_Maritime_continent_average, Height, "g", marker="o",
#         linewidth=4.33, markersize=12)
ax.text(0.05, 0.9, "(d)", transform=ax.transAxes, fontsize=50)
# 尽管我们更改了整个图形的尺寸，但是可以通过以下set_xticks进行x轴标记的手动设置。
ax.set_xticks(np.arange(0, 60, 10))
ax.set_xlim(0, 54.5)
ax.set_ylim(0, 21)
ax.set_ylabel("Height (km)", fontsize=45)
ax.set_xlabel("MaxdBZ (dBZ)", fontsize=40)
ax.tick_params(axis="both", direction='out', which="major", length=10, width=2,
               top=False, right=False)
ax.minorticks_on()
ax.tick_params(axis="both", direction="out", which="minor", length=6, width=1.5,
               top=False, right=False)
# 没有top10%
ax.legend(fontsize=35)
# 加了top10%
# ax.legend(fontsize=30)
# n20_all = data[0]
# n40_all = data[1]
# n20_develop = data1[0]
# n40_develop = data1[1]
# n20_mature = data2[0]
# n40_mature = data2[1]
# n20_dissi = data3[0]
# n40_dissi = data3[1]

# plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.size"] = 30
# fig = plt.figure(1, figsize=(30, 15))
# abcdef = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
# plt.tight_layout()
# plt.title("Different Stages of boxplot", fontsize=40)
# plt.yticks(fontsize=30)
# plt.xticks(fontsize=30)
plt.savefig(f"C:\\Users\\lvyih\\Desktop\\Africa_Americas_Maritime_continent_mdbz.jpeg", bbox_inches="tight", dpi=400)
"""ax.text(
        0.2, 0.1, 'some text',
        horizontalalignment='center',  # 水平居中
        verticalalignment='center',  # 垂直居中
        transform=ax.transAxes  # 使用相对坐标
    )"""
