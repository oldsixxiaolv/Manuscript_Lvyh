# -- coding:utf-8 --
from pyhdf.SD import SD, SDC
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
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
    Volume30 = n30_volume[index_add]
    Volume40 = n40_volume[index_add]
    n20 = n20[index_add]
    n30 = n30[index_add]
    n40 = n40[index_add]
    n20_volume = n20_volume[index_add]
    n40_volume = n40_volume[index_add]
    n30_volume = n30_volume[index_add]
    mdbz = mdbz[index_add]
    return flashrate, maxht20, maxht30, maxht40, maxdbz, minir, npixels_20_R, npixels_30_R, npixels_40_R, n20_volume, n40_volume, n30_volume


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
        "Minir", "D20$_{eq}$", "D30$_{eq}$", "D40$_{eq}$", "Volume20", "Volume40", "Volume30"]
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

labels_pre_mat_post = labels[np.where(labels != 0)]
Maxht40_all = data[8]
FLRate_all = data[0]
df = pd.DataFrame({
    'Maxht40': Maxht40_all,
    'FlRate': FLRate_all,
    'Stage': labels_pre_mat_post
})
df['Stage'] = df['Stage'].astype('category')
print(df.head(20))

# 成熟前期和成熟期的比较
print("成熟前期和成熟期的比较")
stage_pair = [3, 2]
df_pair = df[df['Stage'].isin(stage_pair)].copy()
# 去掉未使用的类别
df_pair['Stage'] = df_pair['Stage'].cat.remove_unused_categories()
# 拟合简化模型（不考虑阶段差异）
model_reduced = ols('FlRate ~ Maxht40', data=df_pair).fit()
# 拟合完整模型（允许阶段有不同的截距和斜率）
model_full = ols('FlRate ~ Stage * Maxht40', data=df_pair).fit()
# 进行 F 检验
anova_result = anova_lm(model_reduced, model_full)
print(anova_result)
# 提取 p 值
p_value = anova_result['Pr(>F)'].iloc[1]
print(f"p-value = {p_value:.4f}")

# 成熟期和成熟后期的比较
print("成熟期和成熟后期的比较")
stage_pair = [2, 1]
df_pair = df[df['Stage'].isin(stage_pair)].copy()
# 去掉未使用的类别
df_pair['Stage'] = df_pair['Stage'].cat.remove_unused_categories()
# 拟合简化模型（不考虑阶段差异）
model_reduced = ols('FlRate ~ Maxht40', data=df_pair).fit()
# 拟合完整模型（允许阶段有不同的截距和斜率）
model_full = ols('FlRate ~ Stage * Maxht40', data=df_pair).fit()
# 进行 F 检验
anova_result = anova_lm(model_reduced, model_full)
print(anova_result)
# 提取 p 值
p_value = anova_result['Pr(>F)'].iloc[1]
print(f"p-value = {p_value:.4f}")

# 成熟前期和成熟后期的比较
print("成熟前期和成熟后期的比较")
stage_pair = [3, 1]
df_pair = df[df['Stage'].isin(stage_pair)].copy()
# 去掉未使用的类别
df_pair['Stage'] = df_pair['Stage'].cat.remove_unused_categories()
# 拟合简化模型（不考虑阶段差异）
model_reduced = ols('FlRate ~ Maxht40', data=df_pair).fit()
# 拟合完整模型（允许阶段有不同的截距和斜率）
model_full = ols('FlRate ~ Stage * Maxht40', data=df_pair).fit()
# 进行 F 检验
anova_result = anova_lm(model_reduced, model_full)
print(anova_result)
# 提取 p 值
p_value = anova_result['Pr(>F)'].iloc[1]
print(f"p-value = {p_value:.4f}")

# 开始进行三阶段整体比较
print("三阶段整体比较")
model_reduced_all = ols('FlRate ~ Maxht40', data=df).fit()
model_full_all = ols('FlRate ~ Stage * Maxht40', data=df).fit()
anova_all = anova_lm(model_reduced_all, model_full_all)
print(anova_all)
p_value_all = anova_all['Pr(>F)'].iloc[1]
print(f"Overall p-value = {p_value_all:.4f}")

