# -- coding:utf-8 --
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from read_shuju import read_shuju
from matplotlib.ticker import ScalarFormatter


def feature_normalize(dt):
    average = np.mean(dt)
    sigma = np.std(dt)
    return (dt - average) / sigma


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
data = read_shuju()
r = data[0]
flashrate = data[1]
npixels20_R = data[2]
npixels30_R = data[3]
npixels40_R = data[4]
npixels40_divide_npixels20 = data[5]
fls20 = data[6]
fls30 = data[7]
fls40 = data[8]
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
flashrate = feature_normalize(flashrate)
npixels20_R = feature_normalize(npixels20_R)
npixels30_R = feature_normalize(npixels30_R)
npixels40_R = feature_normalize(npixels40_R)
npixels40_divide_npixels20 = feature_normalize(npixels40_divide_npixels20)
fls20 = feature_normalize(fls20)
fls30 = feature_normalize(fls30)
fls40 = feature_normalize(fls40)
# viewtime = feature_normalize(viewtime)
# print(len(npixels20_R))
# print(len(npixels40_R))
# print(len(npixels40_divide_npixels20))
# print(len(list(zip(npixels20_R, npixels40_R, npixels40_divide_npixels20))))
# print(len(list(zip(npixels20_R, npixels40_R))))
X = np.array(list(zip(npixels20_R, npixels30_R, npixels40_R
                      ))).reshape(len(npixels20_R), 3)

# 这里得到的distortion是欧几里得距离
distortions = []
# 这里的inertias是样本到其最近聚类中心的平方距离之和
inertias = []
# 轮廓系数是衡量聚类效果好坏的一种评价方式，越接近1表示聚类效果越好
Silhouette_Coefficient = [np.nan]
# CH值越高代表聚类效果也好，这也意味着数据点在聚类间的分布比在聚类内的分布更分散
CH = [np.nan]
K = range(1, 9)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, "euclidean"), axis=1)) / X.shape[0])
    inertias.append(kmeanModel.inertia_)
    if k == 1:
        pass
    else:
        # Silhouette_Coefficient.append(round(float(metrics.silhouette_score(X, kmeanModel.labels_,
        #                                                                    metric="euclidean")), 2))
        CH.append(round(float(metrics.calinski_harabasz_score(X, kmeanModel.labels_))))
        # print(f"k={k}")
        # print("轮廓系数", metrics.silhouette_score(X, kmeanModel.labels_, metric="euclidean"))
        # print("CH", metrics.calinski_harabasz_score(X, kmeanModel.labels_))
        print("---------------------------------")

fig = plt.figure(1, figsize=(4, 3))
# ax = fig.add_subplot(1, 2, 1)
# ax.set_xlabel("k", fontsize=14)
# ax.set_ylabel("distortions", fontsize=14)
# ax.plot(K, distortions, "bx-")
# 设置y轴是科学计数法
y_formatter = ScalarFormatter(useMathText=True)
y_formatter.set_powerlimits((0, 0))
ax1 = fig.add_subplot(1, 1, 1)
ax1.yaxis.set_major_formatter(y_formatter)
ax1.set_xlabel("k", fontsize=14)
ax1.set_ylabel("Inertias", fontsize=14)
ax1.plot(K, inertias, "bx-", color="black")
ax1.set_yticks([50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000])
ax1.set_ylim(10000, 420000)
ax2 = ax1.twinx()
print(CH)
# print(Silhouette_Coefficient)
ax2.yaxis.set_major_formatter(y_formatter)
# 这是设置小于等于或者大于等于啥的数才会按照科学计数法
# y_formatter.set_powerlimits(())
ax2.plot(K, CH, "r^-", color="red")
ax2.set_ylabel("CH", fontsize=14, color="red")
ax2.tick_params(axis="y", colors="red")
ax2.set_yticks([240000, 245000, 250000, 255000, 260000, 265000])
ax2.set_ylim(236500, 267000)
plt.savefig("C:\\Users\\lvyih\\Desktop\\zhoubutu_.jpeg", bbox_inches="tight", dpi=400)

