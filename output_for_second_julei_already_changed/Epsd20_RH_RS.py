# -- coding:utf-8 --
from sklearn.cluster import KMeans, DBSCAN, BisectingKMeans
from sklearn.metrics import adjusted_rand_score
from sklearn import metrics
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from read_shuju import read_shuju
# from read_shuju_add import read_shuju_ADD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def look_for_max(label, k, parameter, coresamples):
    parameter_return = []
    parameter_num = []
    for i in range(k):
        parameter_num.append(len(parameter[coresamples][np.where(label == i)]))
        parameter_return.append(np.max(parameter[coresamples][np.where(label == i)]))
    return parameter_return, parameter_num


def look_for_median(label, k, parameter, coresamples):
    parameter_return = []
    parameter_num = []
    for i in range(k):
        parameter_num.append(len(parameter[coresamples][np.where(label == i)]))
        parameter_return.append(np.median(parameter[coresamples][np.where(label == i)]))
    return parameter_return, parameter_num


def look_for_min(label, k, parameter, coresamples):
    parameter_return = []
    parameter_num = []
    for i in range(k):
        parameter_num.append(len(parameter[coresamples][np.where(label == i)]))
        parameter_return.append(np.min(parameter[coresamples][np.where(label == i)]))
    return parameter_return, parameter_num


# ------------------------------------------------------------------
# 1. 自动计算最优 eps 的函数
# ------------------------------------------------------------------
def find_best_eps(X, min_pts, plot=False):
    """
    给定数据 X 和固定的 min_pts，自动返回最合理的 eps。
    可选画出 k-距离曲线与拐点。
    """
    # 1. 为每个点找最近 min_pts 个邻居的距离
    nbrs = NearestNeighbors(n_neighbors=min_pts + 1, algorithm='kd_tree', n_jobs=-1).fit(X)
    avg_1nn = nbrs.kneighbors(X)[0][:, 1].mean()
    # print('平均 1-NN 距离:', avg_1nn)
    distances, _ = nbrs.kneighbors(X)          #  Shape: (n_samples, min_pts+1)
    k_dist = distances[:, -1]                  # 取第 min_pts 个距离（自己排第一）
    k_dist = np.sort(k_dist)[::-1]             # 从大到小排序，方便找拐点

    x   = np.arange(len(k_dist))
    diff1 = np.gradient(k_dist, x)
    diff2 = np.gradient(diff1, x)
    # print(diff1)
    curv  = np.abs(diff2) / (1 + diff1**2)**1.5
    # print(curv)
    # print(k_dist)
    # print(sorted(list(curv)))
    curv_sort = np.argsort(curv)
    # print(curv[np.argmax(curv)])
    # print(k_dist[np.argmax(curv)])
    # eps_1pc = np.percentile(k_dist, 1)
    # print(eps_1pc)
    # 因为不同系统处理数据的差异，本应该是950的改成949
    knee_idx = curv_sort[-935]
    best_eps = k_dist[curv_sort[-935]]
    if plot:
        plt.figure(figsize=(6, 4))
        plt.plot(x, k_dist, label='k-distance')
        plt.axhline(best_eps, color='r', ls='--', label=f'suggested eps = {best_eps:.2f}')
        plt.axvline(knee_idx, color='r', ls='--', alpha=0.5)
        plt.xlabel('Data points sorted by distance')
        plt.ylabel(f'{min_pts}-distance')
        plt.title('k-distance graph (elbow)')
        plt.legend()
        plt.show()

    return round(best_eps, 2)

def feature_normalize(dt):
    average = np.nanmean(dt[np.isfinite(dt)])
    sigma = np.nanstd(dt[np.isfinite(dt)])
    return (dt - average) / sigma


def feature_normalize_back(center_clusters, *former_data):
    back_matrix1 = []
    back_matrix2 = []
    for i in range(len(former_data)):
        average = np.nanmean(former_data[i][np.isfinite(former_data[i])])
        sigma = np.nanstd(former_data[i][np.isfinite(former_data[i])])
        back_matrix1.append(sigma)
        back_matrix2.append(average)
    # average1 = np.nanmean(former_data1[np.isfinite(former_data1)])
    # average2 = np.nanmean(former_data2[np.isfinite(former_data2)])
    # average3 = np.nanmean(former_data3[np.isfinite(former_data3)])
    # average4 = np.nanmean(former_data4[np.isfinite(former_data4)])
    # sigma1 = np.nanstd(former_data1[np.isfinite(former_data1)])
    # sigma2 = np.nanstd(former_data2[np.isfinite(former_data2)])
    # sigma3 = np.nanstd(former_data3[np.isfinite(former_data3)])
    # sigma4 = np.nanstd(former_data4[np.isfinite(former_data4)])
    back_matrix1 = np.array(back_matrix1)
    back_matrix2 = np.array(back_matrix2)
    return center_clusters * back_matrix1 + back_matrix2


def converge(data, data_add):
    data_mid = []
    for i, j in zip(data, data_add):
        data_mid.append(np.concatenate((i, j)))
    return data_mid


def for_(data, stage):
    mid = []
    for i in data:
        mid.append(i[stage])
    return mid


def duqu_excel_julei(path):
    import pandas as pd
    file_path = path
    df = pd.read_excel(file_path, usecols=[1], names=None)
    dali = df.values.tolist()
    result = []
    for i in dali:
        result.append(i[0])
    return result


def other_parameters(label, k, parameter, coresamples):
    parameter_return = []
    parameter_num = []
    for i in range(k):
        parameter_num.append(len(parameter[coresamples][np.where(label == i)]))
        parameter_return.append(np.mean(parameter[coresamples][np.where(label == i)]))
    return list(map(float, parameter_return)), parameter_num


def other_parameters_output_all(label, k, parameter, coresamples):
    parameter_return = []
    parameter_num = []
    w = 0
    for i in range(k):
        parameter_num.append(len(parameter[coresamples][np.where(label == i)]))
        parameter_return.append(parameter[coresamples][np.where(label == i)])
    for j in parameter_return:
        w += 1
        print(f"The {w} cluster")
        print(sorted(list(map(float, j))))


def find_clusters(core_samples, label, k):
    return_data = []
    for i in range(k):
        return_samples = core_samples[np.where(label == i)]
        return_data.append(return_samples)
    return return_data


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
data_last = read_shuju()
# stage = duqu_excel_julei(r"C:\Users\lvyih\Desktop\stage.xlsx")
# data_last = for_(data_last, stage)
# stage_2 = duqu_excel_julei(r"C:\Users\lvyih\Desktop\stage_2.xlsx")
# data_last = for_(data_last, stage)
# data_last = for_(data_last, stage_2)
# data_add = read_shuju_ADD()
# data_last = converge(data, data_add)
# r, maxht20, maxht30, maxht40, npixels_20, npixels_30, npixels_40, flash_20, flash_30, flash_40, minir, flashrate
r = data_last[0]
print(len(r))
maxht20 = data_last[1]
maxht30 = data_last[2]
maxht40 = data_last[3]
npixels_20 = data_last[4]
npixels_30 = data_last[5]
npixels_40 = data_last[6]
flash_20 = data_last[7]
flash_30 = data_last[8]
flash_40 = data_last[9]
minir = data_last[10]
flashrate = data_last[11]
maxht20_minux_maxht40 = data_last[12]
ellip_20 = data_last[13]
ellip_30 = data_last[14]
ellip_40 = data_last[15]
maxdbz = data_last[16]
maxht = data_last[17]
npx40_divide_npx20 = data_last[18]
npx40_divide_npx30 = data_last[19]
ellip_20_maxht20 = data_last[20]
ellip_30_maxht30 = data_last[21]
ellip_40_maxht40 = data_last[22]
Volume20 = data_last[23]
Volume40 = data_last[24]
# print(len(viewtime))
# viewtime = viewtime[~np.isnan(npixels40_divide_npixels20)]
# npixels20_R = npixels20_R[~np.isnan(npixels40_divide_npixels20)]
# npixels40_R = npixels40_R[~np.isnan(npixels40_divide_npixels20)]
# npixels40_divide_npixels20 = npixels40_divide_npixels20[~np.isnan(npixels40_divide_npixels20)]
# index1 = list(np.where(viewtime < 190))
# viewtime = viewtime[index1[0]]
# print(len(viewtime))
# npixels20_R = npixels20_R[index1[0]]
# npixels40_R = npixels40_R[index1[0]]
# npixels40_divide_npixels20 = npixels40_divide_npixels20[index1[0]]
r = feature_normalize(r)
maxht40_divide_maxht20 = feature_normalize(maxht40 / maxht20)
maxht20_minus_maxht40_divide_maxht20 = feature_normalize((maxht20-maxht40) / maxht20)
maxht20 = feature_normalize(maxht20)
maxht30 = feature_normalize(maxht30)
maxht40 = feature_normalize(maxht40)
npixels_20 = feature_normalize(npixels_20)
npixels_30 = feature_normalize(npixels_30)
npixels_40 = feature_normalize(npixels_40)
flash_20 = feature_normalize(flash_20)
flash_30 = feature_normalize(flash_30)
flash_40 = feature_normalize(flash_40)
minir = feature_normalize(minir)
flashrate = feature_normalize(flashrate)
maxht20_minux_maxht40 = feature_normalize(maxht20_minux_maxht40)
ellip_20 = feature_normalize(ellip_20)
ellip_30 = feature_normalize(ellip_30)
ellip_40 = feature_normalize(ellip_40)
maxdbz = feature_normalize(maxdbz)
maxht = feature_normalize(maxht)
npx40_divide_npx20 = feature_normalize(npx40_divide_npx20)
npx40_divide_npx30 = feature_normalize(npx40_divide_npx30)
ellip_20_maxht20 = feature_normalize(ellip_20_maxht20)
ellip_30_maxht30 = feature_normalize(ellip_30_maxht30)
ellip_40_maxht40 = feature_normalize(ellip_40_maxht40)
Volume40_divide_20 = feature_normalize(Volume40 / Volume20)
Volume20 = feature_normalize(Volume20)
Volume40 = feature_normalize(Volume40)
np.set_printoptions(threshold=np.inf)
# viewtime = feature_normalize(viewtime)
# print(len(npixels20_R))
# print(len(npixels40_R))
# print(len(npixels40_divide_npixels20))
# print(len(list(zip(npixels20_R, npixels40_R, npixels40_divide_npixels20))))
# print(len(list(zip(npixels20_R, npixels40_R))))
X = np.array(list(zip(ellip_20, maxht20_minus_maxht40_divide_maxht20, npx40_divide_npx20
# Volume40_divide_20
                      ))).reshape(len(ellip_20), 3)
# print('各维度极差:', np.ptp(X, axis=0))
# print('各维度 std :', X.std(axis=0))
# print('各维度均值:', X.mean(axis=0))
best_eps = find_best_eps(X, 4)
print(best_eps)
# 这里得到的distortion是欧几里得距离
distortions = []
# 这里的inertias是样本到其最近聚类中心的平方距离之和
inertias = []
DBI = []
Silhouette_Coefficient = [np.nan]
CH = [np.nan]
K = range(1, 8)
for k in K:
    dbscan = DBSCAN(eps=best_eps, min_samples=4)
    labels = dbscan.fit_predict(X)
    core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_samples_mask[dbscan.core_sample_indices_] = True
    core_samples = X[core_samples_mask]
    # print(len(core_samples))

    """
    if k==1:
        pass
    else:
        w_cand = np.linspace(0.2, 2.0, 9)                 # 0.2–3.0 等分 15 档
        best_w, best_sil = None, 3
        baby = 0
        for w1 in w_cand:                                  # 3 维暴力 15³ ≈ 3375 次
            for w2 in w_cand:
                for w3 in w_cand:
                    X_w = core_samples * [w1, w2, w3]
                    # change to BesectingKMeans
                    # lab = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=100, tol=1e-5).fit_predict(X_w)
                    kmean_fit = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=100, tol=1e-5).fit(X_w)
                    CH_sil = round(float(metrics.calinski_harabasz_score(X_w, kmean_fit.labels_)))
                    # sil = metrics.davies_bouldin_score(X_w, lab)
                    print(f"{baby}")
                    baby += 1
                    if CH_sil > best_sil:
                        best_sil, best_w = CH_sil, [w1, w2, w3]
        print("最优权重:", best_w, "CH:", best_sil)
        core_samples = core_samples * best_w
    """

    kmeanModel = BisectingKMeans(n_clusters=k, init="k-means++", n_init=100, max_iter=10000 , tol=1e-8).fit(core_samples)
    labels = kmeanModel.labels_
    print(len(labels))
    cluster_centers = feature_normalize_back(kmeanModel.cluster_centers_, data_last[13], data_last[12]/data_last[1], data_last[18]
#data_last[24]/data_last[23]
)
    print(cluster_centers)
    # print(labels)
    Epsd = find_clusters(core_samples[:, 0], labels, k)
    RH = find_clusters(core_samples[:, 1], labels, k)
    RS = find_clusters(core_samples[:, 2], labels, k)
    kmean2_DBI = BisectingKMeans(n_clusters=k, init="k-means++").fit_predict(core_samples)
    # distortions.append(sum(np.min(cdist(core_samples, kmeanModel.cluster_centers_, "euclidean"), axis=1)) / core_samples.shape[0])
    inertias.append(kmeanModel.inertia_)
    if k == 1:
        pass
    else:
        # Silhouette_Coefficient.append(round(float(metrics.silhouette_score(X, kmeanModel.labels_,
        #                                                          metric="euclidean")), 2))
        dbi = metrics.davies_bouldin_score(core_samples, kmean2_DBI)
        CH.append(round(float(metrics.calinski_harabasz_score(core_samples, kmeanModel.labels_))))
        DBI.append(dbi)
        print(f"k={k}")
        print("DBI", dbi)
        # other_parameters_output_all(labels, k, data_last[0], core_samples_mask)
        print("r", other_parameters(labels, k, data_last[0], core_samples_mask))
        print("r_max", look_for_max(labels, k, data_last[0], core_samples_mask))
        print("r_median", look_for_median(labels, k, data_last[0], core_samples_mask))
        print("r_min", look_for_min(labels, k, data_last[0], core_samples_mask))
        print("maxht20", other_parameters(labels, k, data_last[1], core_samples_mask))
        print("maxht20_max", look_for_max(labels, k, data_last[1], core_samples_mask))
        print("maxht20_median", look_for_median(labels, k, data_last[1], core_samples_mask))
        print("maxht20_min", look_for_min(labels, k, data_last[1], core_samples_mask))
        print("maxht30", other_parameters(labels, k, data_last[2], core_samples_mask))
        print("maxht30_max", look_for_max(labels, k, data_last[2], core_samples_mask))
        print("maxht30_median", look_for_median(labels, k, data_last[2], core_samples_mask))
        print("maxht30_min", look_for_min(labels, k, data_last[2], core_samples_mask))
        print("maxht40", other_parameters(labels, k, data_last[3], core_samples_mask))
        print("maxht40_max", look_for_max(labels, k, data_last[3], core_samples_mask))
        print("maxht40_median", look_for_median(labels, k, data_last[3], core_samples_mask))
        print("maxht40_min", look_for_min(labels, k, data_last[3], core_samples_mask))
        print("Area20", other_parameters(labels, k, data_last[4], core_samples_mask))
        print("Area30", other_parameters(labels, k, data_last[5], core_samples_mask))
        print("Area40", other_parameters(labels, k, data_last[6], core_samples_mask))
        print("Epsd20", other_parameters(labels, k, data_last[13], core_samples_mask))
        print("Epsd30", other_parameters(labels, k, data_last[14], core_samples_mask))
        print("Epsd40", other_parameters(labels, k, data_last[15], core_samples_mask))
        print("Flashrate", other_parameters(labels, k, data_last[11], core_samples_mask))
        print("Fls40", other_parameters(labels, k, data_last[9], core_samples_mask))
        print("Maxht20-Maxht40", other_parameters(labels, k, data_last[12], core_samples_mask))
        print("Volume40/Volume20", other_parameters(labels, k, data_last[24]/data_last[23], core_samples_mask))
        print("Epsd20_maxht20", other_parameters(labels, k, data_last[20], core_samples_mask))
        print("Epsd30_maxht30", other_parameters(labels, k, data_last[21], core_samples_mask))
        print("Epsd40_maxht40", other_parameters(labels, k, data_last[22], core_samples_mask))
        # print("Flashrate", other_parameters(labels, k, data_last[11], core_samples_mask))
        # print("轮廓系数", metrics.silhouette_score(core_samples, kmeanModel.labels_, metric="euclidean"))
        # print("CH", metrics.calinski_harabasz_score(X, kmeanModel.labels_))
        print("---------------------------------")


fig = plt.figure(1, figsize=(12, 4))
# ax = fig.add_subplot(1, 2, 1)
# ax.set_xlabel("k", fontsize=14)
# ax.set_ylabel("distortions", fontsize=14)
# ax.plot(K, distortions, "bx-")
ax1 = fig.add_subplot(1, 3 ,1)
ax1.set_xlabel("k", fontsize=14)
ax1.set_ylabel("Inertias", fontsize=14)
ax1.plot(K, inertias, "bx-")
ax2 = ax1.twinx()
print(CH)
ax2.plot(K, CH, "r^-")
ax2.tick_params(axis="y", colors="red")
ax2.set_ylabel("CH", fontsize=14, color="red")
ax3 = fig.add_subplot(1, 3, 2, projection="3d")
ax3.scatter(maxht20_minus_maxht40_divide_maxht20, npx40_divide_npx20, ellip_30, c="blue", marker="o", s=0.1)
ax4 = fig.add_subplot(1, 3, 3, projection="3d")
ax4.scatter(RH[0], RS[0], Epsd[0], c="red", marker="x", s=0.1)
ax4.scatter(RH[1], RS[1], Epsd[1], c="blue", marker="o", s=0.1)
ax4.scatter(RH[2], RS[2], Epsd[2], c="green", marker="s", s=0.1)
# ax4.scatter(RH[3], RS[3], Epsd[3], c="brown", marker="s", s=0.1)
# ax4.scatter(RH[4], RS[4], Epsd[4], c="blueviolet", marker="s", s=0.1)
plt.savefig("/root/git/Project_develop/manuscript_review/figures/zhoubutu____.jpeg", bbox_inches="tight", dpi=400)
np.save("./core_samples.npy", core_samples_mask)
np.save("./lables.npy", labels)







r = data_last[0]
maxht20 = data_last[1]
maxht30 = data_last[2]
maxht40 = data_last[3]
npixels_20 = data_last[4]
npixels_30 = data_last[5]
npixels_40 = data_last[6]
flash_20 = data_last[7]
flash_30 = data_last[8]
flash_40 = data_last[9]
minir = data_last[10]
flashrate = data_last[11]
maxht20_minux_maxht40 = data_last[12]
ellip_20 = data_last[13]
ellip_30 = data_last[14]
ellip_40 = data_last[15]
maxdbz = data_last[16]
maxht = data_last[17]
npx40_divide_npx20 = data_last[18]
npx40_divide_npx30 = data_last[19]
ellip_20_maxht20 = data_last[20]
ellip_30_maxht30 = data_last[21]
ellip_40_maxht40 = data_last[22]
Volume20 = data_last[23]
Volume40 = data_last[24]
df = DataFrame({"r": r[core_samples_mask], "maxht20": maxht20[core_samples_mask],
                "maxht30": maxht30[core_samples_mask], "maxht40": maxht40[core_samples_mask],
                "npixels20_R": npixels_20[core_samples_mask], "npixels30_R": npixels_30[core_samples_mask], "npixels40_R": npixels_40[core_samples_mask],
                "flash20": flash_20[core_samples_mask], "flash30": flash_30[core_samples_mask],
                "flash40": flash_40[core_samples_mask], "minir": minir[core_samples_mask],
                "flashrate": flashrate[core_samples_mask], "maxht20_minux_maxht40": maxht20_minux_maxht40[core_samples_mask],
                "ellip20": ellip_20[core_samples_mask], "ellip30": ellip_30[core_samples_mask], "ellip40": ellip_40[core_samples_mask],
                "maxdbz": maxdbz[core_samples_mask], "maxht": maxht[core_samples_mask],
                "npx40_divide_npx20": npx40_divide_npx20[core_samples_mask], "npx40_divide_npx30": npx40_divide_npx30[core_samples_mask],
                "ellip_20_maxht20": ellip_20_maxht20[core_samples_mask], "ellip_30_maxht30": ellip_30_maxht30[core_samples_mask],
                "ellip_40_maxht40": ellip_40_maxht40[core_samples_mask], "Volume20": Volume20[core_samples_mask], "Volume40": Volume40[core_samples_mask],
                "labels": labels})
df.to_excel(r"./ouput_for_julei.xlsx", sheet_name="sheet1", index=False)
