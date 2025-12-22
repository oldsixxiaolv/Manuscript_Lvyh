# -- coding:utf-8 --
from read_shuju import read_shuju
from pandas import DataFrame
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['font.family'] = "Times New Roman"  # 设置中文
plt.rcParams["font.size"] = 15
all_list = read_shuju()
r = all_list[0]
npixels20_R = all_list[1]
npixels40_R = all_list[2]
npixels40_divide_npixels20 = all_list[3]
flashcount = all_list[4]
fig = plt.figure(figsize=(15, 15))
ax = plt.axes(projection="3d")
ax.scatter3D(npixels20_R, npixels40_R, npixels40_divide_npixels20, s=0.01, color="red", marker="^")
ax.set_xlabel('npx20R')
ax.set_ylabel("npx40R")
ax.set_zlabel('npx20/40')
plt.savefig(f"C:\\Users\\lvyih\\Desktop\\kongjian_sandian.jpeg", dpi=400)
# df = DataFrame({"r": all_list[0], "npixels_20_R": all_list[1],
#                 "npixels_40_R": all_list[2], "npixels40_divide_npixels20": all_list[3], "flashcount": all_list[4]})
# df.to_excel(r"C:\Users\lvyih\Desktop\x.xlsx", sheet_name="sheet1", index=False)
