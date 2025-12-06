# -- coding:utf-8 --
from shuju_process import read_shuju
import numpy as np
from tabulate import tabulate


"""制造索引"""
data = read_shuju()
hh = 0
table_output_mid = []
table_output = [["parameter_name", "developing_stage", "mature_stage", "dissipating_stage"]]
"""提取数据"""
name = ['flashrate/fl min-1', 'maxht/km', 'maxht20/km', 'maxht30/km', 'maxht40/km',
        'minir/K', 'maxnsrain', "npixels_40_area/km2",
        "npixels_20_area/km2", "npixels_30_area/km2", "maxdbz20-maxdbz40/km",
        "ellipsoidal_ratio_20/km-1", "ellipsoidal_ratio_30/km-1",
        "ellipsoidal_ratio_40/km-1", "flashrate_density_20dbz/min-1 km-2 (100 fl)",
        "flashrate_density_30dbz/min-1 km-2 (100 fl)", "flashrate_density_40dbz/min-1 km-2 (100 fl)"]
# 可以让python不省略输出
np.set_printoptions(threshold=np.inf)
"""主程序"""
for i in ["developing_stage", "mature_stage", "dissipating_stage"]:
    mid = []
    for j in data:
        r = j[0]
        index1 = np.where(r <= 1)
        index2 = np.where(r > 0.96)
        index3 = np.where(r <= 0.96)
        index4 = np.where(r > 0.9)
        index5 = np.where(r <= 0.9)
        index6 = np.where(r > 0.54)
        index_developing = list(set(index1[0]) & set(index2[0]))
        index_maturity = list(set(index3[0]) & set(index4[0]))
        index_dissipation = list(set(index5[0]) & set(index6[0]))
        indexx = [index_developing, index_maturity, index_dissipation]
        index = indexx[hh]
        midmid = j[1][index]
        mid.append(str(np.round(np.nanmean(midmid[np.isfinite(midmid)]), 2)))
    hh += 1
    table_output_mid.append(mid)
"""准备制作图表"""
a = np.array(table_output_mid)
k = 0
for j in list(map(list, a.T)):
    b = [name[k]] + j
    table_output.append(b)
    k += 1
print(tabulate(table_output, headers="firstrow"))


