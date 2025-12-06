# -- coding:utf-8 --
from sklearn.cluster import KMeans, DBSCAN
from sklearn import metrics
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from read1 import read_shuju1
from sklearn.neighbors import NearestNeighbors
from read2 import read_shuju2
from sklearn.preprocessing import StandardScaler


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
data_last = read_shuju1()
# stage = duqu_excel_julei(r"C:\Users\lvyih\Desktop\stage.xlsx")
# data_last = for_(data_last, stage)
# stage_2 = duqu_excel_julei(r"C:\Users\lvyih\Desktop\stage_2.xlsx")
# data_last = for_(data_last, stage)
# data_last = for_(data_last, stage_2)
# data_add = read_shuju_ADD()
# data_last = converge(data, data_add)
# r, maxht20, maxht30, maxht40, npixels_20, npixels_30, npixels_40, flash_20, flash_30, flash_40, minir, flashrate
r1 = data_last[0]
maxht20_1 = data_last[1]
maxht30_1 = data_last[2]
maxht40_1 = data_last[3]
Area20_1 = data_last[4]
Area30_1 = data_last[5]
Area40_1 = data_last[6]
flashrate_1 = data_last[11]
fls40_1 = data_last[9]
print(len(r1))
data_last2 = read_shuju2()
r2 = data_last2[0]
maxht20_2 = data_last2[1]
maxht30_2 = data_last2[2]
maxht40_2 = data_last2[3]
Area20_2 = data_last2[4]
Area30_2 = data_last2[5]
Area40_2 = data_last2[6]
flashrate_2 = data_last2[11]
fls40_2 = data_last2[9]
print(len(r2))
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
np.set_printoptions(threshold=np.inf)
def return_max_mean_median(parameter):
    return float(np.max(parameter)), float(np.mean(parameter)), float(np.median(parameter)), float(np.min(parameter))
# print(r1)
print("without 40dBZ but with flash detected")
print("max , mean, median,  min    r", return_max_mean_median(r1))
print("max , mean, median,  min    maxht20", return_max_mean_median(maxht20_1))
print("max , mean, median,  min    maxht30", return_max_mean_median(maxht30_1))
print("max , mean, median,  min    maxht40", return_max_mean_median(maxht40_1))
print("max , mean, median,  min    Area20", return_max_mean_median(Area20_1))
print("max , mean, median,  min    Area30", return_max_mean_median(Area30_1))
print("max , mean, median,  min    Area40", return_max_mean_median(Area40_1))
print("max , mean, median,  min    Flashrate", return_max_mean_median(flashrate_1))
print("max , mean, median,  min    Fls_40", return_max_mean_median(fls40_1))

# print(r2)
print("with 40dBZ but without flash detected")
print("max , mean, median,  min    r", return_max_mean_median(r2))
print("max , mean, median,  min    maxht20", return_max_mean_median(maxht20_2))
print("max , mean, median,  min    maxht30", return_max_mean_median(maxht30_2))
print("max , mean, median,  min    maxht40", return_max_mean_median(maxht40_2))
print("max , mean, median,  min    Area20", return_max_mean_median(Area20_2))
print("max , mean, median,  min    Area30", return_max_mean_median(Area30_2))
print("max , mean, median,  min    Area40", return_max_mean_median(Area40_2))
print("max , mean, median,  min    Flashrate", return_max_mean_median(flashrate_2))
print("max , mean, median,  min    Fls_40", return_max_mean_median(fls40_2))
# plt.boxplot(r2)
# plt.savefig("./r_boxplot.jpeg")
