import matplotlib.pyplot as plt
from all_the_shuju_two import allshuju_two
from read_shuju import read_shuju
from x_labels import x_labels, x_labels_for_anix
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch

def round2(x):
    """由于之后要用到map函数，在此创建一个函数"""
    ok = round(x, 1)
    return ok


def researching(x, y, fre, begin=0):
    """用来找到0.9和0.8的阙值x"""
    jiu = 0
    for w in fre:
        print(w)
        if (w > x) and (w < y):
            jiu = round(w, 2)
            break
        begin += 0.01
    return round(begin, 2), jiu


def researching_num(num, fre, frequence, begin=0):
    """用来找到数量小于num的r"""
    jiu = 0
    for (w, q) in zip(fre, frequence):
        print(w)
        if w < num:
            jiu = round(q, 2)
            break
        begin += 0.01
    begin -= 0.01
    return round(begin, 2), jiu


def researching_Ratio(Ratio, fre):
    Ratio_mid = round((1 - Ratio) / 0.01)
    return fre[Ratio_mid]


def for_(data, stage):
    return data[stage]


def duqu_excel_julei(path):
    import pandas as pd
    file_path = path
    df = pd.read_excel(file_path, usecols=[1], names=None)
    dali = df.values.tolist()
    result = []
    for i in dali:
        result.append(i[0])
    return result


"""读取数据部分"""
all_list = read_shuju()
# stage = duqu_excel_julei(r"C:\Users\lvyih\Desktop\stage.xlsx")
# all_list = for_(all_list, stage)
# all_list2 = read_shuju_ADD()
r = all_list
# r2 = all_list2
# r = np.concatenate((r, r2))
# print(len(r))
# r中有一个空缺值，通过allshuju_two处理后就会消去。
# 这里的frequency最终得到的是0-1的每隔0.01的个数作为元素的列表
frequency = []
allshuju_two(r, frequency)
hh = 0
for i in frequency:
    hh += i
print(hh)
print(len(r))
# 这里的cumsum进行一个累加操作
frequence = np.cumsum(frequency)
# 这里进行求比例处理
frequence = np.divide(frequence, len(r))
# 做到0.96的界限
# begin2, jiu2 = researching(0.959, 0.961, frequence)
# 做到0.95界限
# begin2, jiu2 = researching(0.948, 0.952, frequence)
# 做到0.9的界限
begin3, jiu3 = researching(0.895, 0.905, frequence)
begin1, jiu1 = researching_num(100, frequency, frequence)
begin2 = 0.79
jiu2 = researching_Ratio(begin2, frequence)
print(begin1)
print(begin2)
label = x_labels()
labels_for_anix = x_labels_for_anix()
labels = []
label = list(map(str, label))
for n in range(0, 101):
    mid = n / 100
    labels.append(float(f"{mid:.2f}"))
label[-1] = "0"
"""绘制图形部分"""
# 我们要注意的是如果我们要对x轴进行自定义然后又要在画图了之后进行设置，一定要用fig，ax=plt.subplots()这个函数才行
fig, ax2 = plt.subplots(1, 1, figsize=(40, 16.75))
# 设置全局的字体样式（即更改默认值）[只对最外层的有作用]
# plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.size"] = 28
# set_xticks和set_xticklabels一般要一起出现，xticklabels对xticks的进行修正。
# (xticks只输入列表或者数组等，后不加参数，参数在xticklabels那里设置)
# bar函数可以让x轴是数字或者就是字符串
# 我们总结一点就是如果我们画的图的x轴就是数字，那么一定要先用set_xticks先来操作再用set_xticklabels
# 如果我们画的图的x轴就是字符串的，那么set_xticklabels就可以直接用
"""先绘制第一层图形"""
# 这里的plot加上一个"r"的意思是指明线是红色的
plt.rc('axes', linewidth=3)
ax2.plot(labels, frequence)
ax2.fill_between(labels, frequence, hatch="|", color="#D7F4C1")
ax2.set_yticks(np.array(list(map(round2, list(np.arange(0, 1.1, 0.2))))))
ax2.set_ylabel("percent (%)", font={"family": "Times New Roman", "size": 75})
ax2.set_yticklabels(np.array(list(map(round2, list(np.arange(0, 110, 20))))),
                    font={"family": "Times New Roman", "size": 55})
ax2.set_xticks(labels)
ax2.set_xlabel("Ratio", font={"family": "Times New Roman", "size": 75})
ax2.set_xticklabels(label, font={"family": "Times New Roman", "size": 55})
# ax2.axis(labels.extend(list(map(round2, list(np.arange(0, 1.1, 0.2))))), fontsize=16)
# ax2.vlines(begin1, 0, jiu1, linestyles="dashed", color="red", label=f"r={round(1-begin1, 2)} ,  95%", linewidth=4)
# ax2.hlines(jiu1, begin1, 1, linestyles="dashed", colors="red", linewidth=4)
ax2.vlines(1-begin2, 0, jiu2, linestyles="dashed", color="black",
           label=f"Ratio = {round(begin2, 2)} ,  {round(jiu2 * 100)}%", linewidth=4)
ax2.hlines(jiu2, 1-begin2, 1, linestyles="dashed", color="black", linewidth=4)
# ax2.vlines(begin3, 0, jiu3, linestyles="dashed", color="black",
#            label=f"Ratio = {round(1-begin3, 2)} ,  {round(jiu3 * 100)}%", linewidth=4)
# ax2.hlines(jiu3, begin3, 1, linestyles="dashed", color="black", linewidth=4)
# ax2.hlines(jiu1, 0, 1, linestyles="dashed", color="blue", linewidth=4)
# 指明x和y的范围
ax2.set_ylim(-0.01, None)
# 让r的坐标从最左边的哪里开始出现标签
ax2.set_xlim(-0.012, 1.007)
# 指明图的标签所在的位置
# ax2.legend(loc="lower right", prop={"family": "Times New Roman", "size": 63})
# ax2.set_title("TRMM tropical convective dataset\n"
#               "The distribution of different values of convective precipitation proportions",
#               font={"family": "Times New Roman", "size": 65})
ax2.get_yaxis().set_visible(False)
ax2.tick_params(axis="x", which="major", length=7.5, width=3, pad=10)
# 手动调整一下x轴的刻度值的长度
xticks = ax2.xaxis.get_major_ticks()
for i in range(0, 101, 5):
    xticks[i].tick1line.set_markersize(15)
# ax2.tick_params(axis="x", which="minor", length=8, width=2)
# ax2.yaxis.tick_right()

"""从这里绘制第二层图形"""
ax = ax2.twinx()
ax.set_ylabel("Frequence (#)", font={"family": "Times New Roman", "size": 75})
ax.set_yticks(np.array(list(map(int, list(np.arange(0, 5500, 1000))))))
ax.set_yticklabels(np.array(list(map(int, list(np.arange(0, 5500, 1000)))))
                   , font={"family": "Times New Roman", "size": 55})
ax.bar(labels, frequency, width=0.006)
ax.set_ylim(-40, 5500)  # (-40, None)
ax.yaxis.set_label_position("left")
ax.tick_params(axis="y", which="major", length=15, width=3, pad=10)
# ax.hlines(100, 0, 1, linestyles="dashed", color="black", linewidth=4)
# ax.tick_params(axis="y", direction='in', which="minor", length=12, width=2)
# ax.yaxis.tick_right()
# 设置让y轴不显示
# ax.get_yaxis().set_visible(False)
"""第三层画图"""
ax1 = ax.twinx()
ax1.plot(labels, frequence, color="red", linewidth=5)
ax1.set_yticks(np.array(list(map(round2, list(np.arange(0, 1.1, 0.2))))))
ax1.set_yticklabels(np.array(list(map(round2, list(np.arange(0, 110, 20))))),
                    font={"family": "Times New Roman", "size": 55})
ax1.set_ylabel("CDF (%)", font={"family": "Times New Roman", "size": 75}, color="red")
ax1.set_ylim(-0.01, None)
# y轴的刻度的长度和宽度进行设置
ax1.tick_params(axis="y", which="major", length=15, width=3, pad=10, colors="red")
ax1.vlines(1-begin2, 0, jiu2, linestyles="dashed", color="black",
            label=f"Ratio = {round(begin2, 2)} ,  {round(jiu2*100)}%", linewidth=4)
ax1.hlines(jiu2, 1-begin2, 1, linestyles="dashed", color="black", linewidth=4)
# ax1.tick_params(axis="y", direction='in', which="minor", length=12, width=2)
# 储存绘制的图形
# 把ax2再画蓝线
ax2.legend(loc=(0.66, 0.75), prop={"family": "Times New Roman", "size": 63})
# 嵌入局部放大图坐标系
# width和height是子坐标系的宽度和高度，loc是子坐标系的位置，bbox_to_anchor是边界框, bbox_transform从父坐标系到子坐标系的几何映射
axin = inset_axes(ax2, width="100%", height="100%", bbox_to_anchor=(0.7941, 0.12, 0.201, 0.45), bbox_transform=ax2.transAxes)
axin.plot(labels, frequence)
axin.fill_between(labels, frequence, hatch="|", color="#D7F4C1")
axin.set_yticks(np.array(list(map(round2, list(np.arange(0, 1.1, 0.2))))))
# axin.set_ylabel("percent (%)", font={"family": "Times New Roman", "size": 75})
axin.set_yticklabels(np.array(list(map(round2, list(np.arange(0, 110, 20))))),
                    font={"family": "Times New Roman", "size": 55})
axin.set_xticks(labels)
# axin.set_xlabel("Ratio", font={"family": "Times New Roman", "size": 75})
axin.set_xticklabels(label, font={"family": "Times New Roman", "size": 55})
# ax2.axis(labels.extend(list(map(round2, list(np.arange(0, 1.1, 0.2))))), fontsize=16)
# ax2.vlines(begin1, 0, jiu1, linestyles="dashed", color="red", label=f"r={round(1-begin1, 2)} ,  95%", linewidth=4)
# ax2.hlines(jiu1, begin1, 1, linestyles="dashed", colors="red", linewidth=4)
# axin.vlines(1-begin2, 0, jiu2, linestyles="dashed", color="blue",
#            label=f"Ratio = {round(begin2, 2)} ,  {round(jiu2 * 100)}%", linewidth=4)
# axin.hlines(jiu2, 1-begin2, 1, linestyles="dashed", color="blue", linewidth=4)
# ax2.hlines(jiu1, 0, 1, linestyles="dashed", color="blue", linewidth=4)
# 指明x和y的范围
axin.set_ylim(-0.01, None)
# 让r的坐标从最左边的哪里开始出现标签
axin.set_xlim(-0.012, 1.005)
# 指明图的标签所在的位置
# ax2.legend(loc="lower right", prop={"family": "Times New Roman", "size": 63})
# ax2.set_title("TRMM tropical convective dataset\n"
#               "The distribution of different values of convective precipitation proportions",
#               font={"family": "Times New Roman", "size": 65})
axin.get_yaxis().set_visible(False)
axin.tick_params(axis="x", which="major", length=7.5, width=3, pad=10)
# 手动调整一下x轴的刻度值的长度
xticks = axin.xaxis.get_major_ticks()
for i in range(0, 101, 5):
    xticks[i].tick1line.set_markersize(15)
# ax2.tick_params(axis="x", which="minor", length=8, width=2)
# ax2.yaxis.tick_right()

"""从这里绘制第二层图形"""
axin1 = axin.twinx()
# axin1.set_ylabel("Frequence (#)", font={"family": "Times New Roman", "size": 75})
axin1.set_yticks(np.array(list(map(int, list(np.arange(0, 5500, 25))))))
axin1.set_yticklabels(np.array(list(map(int, list(np.arange(0, 5500, 25)))))
                   , font={"family": "Times New Roman", "size": 55})
axin1.bar(labels, frequency, width=0.006)
# axin1.set_ylim(-40, 5500)  # (-40, None)
axin1.yaxis.set_label_position("left")
axin1.tick_params(axis="y", which="major", length=15, width=3, pad=10)
# ax.hlines(100, 0, 1, linestyles="dashed", color="black", linewidth=4)
# ax.tick_params(axis="y", direction='in', which="minor", length=12, width=2)
# ax.yaxis.tick_right()
# 设置让y轴不显示
axin.get_yaxis().set_visible(False)
"""第三层画图"""
axin2 = axin1.twinx()
axin2.plot(labels, frequence, color="red", linewidth=5)
axin2.set_yticks(np.array(list(map(round2, list(np.arange(0, 1.1, 0.2))))))
axin2.set_yticklabels(np.array(list(map(round2, list(np.arange(0, 110, 20))))),
                    font={"family": "Times New Roman", "size": 55})
axin2.get_yaxis().set_visible(False)
# axin2.set_ylabel("CDF (%)", font={"family": "Times New Roman", "size": 75}, color="red")
# axin2.set_ylim(-0.01, None)
# y轴的刻度的长度和宽度进行设置
# axin2.tick_params(axis="y", which="major", length=15, width=3, pad=10, colors="red")
# axin2.vlines(1-begin2, 0, jiu2, linestyles="dashed", color="blue",
#             label=f"Ratio = {round(begin2, 2)} ,  {round(jiu2*100)}%", linewidth=4)
# axin2.hlines(jiu2, 1-begin2, 1, linestyles="dashed", color="blue", linewidth=4)
# ax1.tick_params(axis="y", direction='in', which="minor", length=12, width=2)
# 储存绘制的图形
# 把ax2再画蓝线
# axin.legend(loc=(0.66, 0.8), prop={"family": "Times New Roman", "size": 63})
# zone_left = 0.8
# zone_right = 1
axin.set_xlim(0.795, 1)
axin.set_ylim(0, 0.4)
# axin.get_xaxis().set_visible(False)
axin1.set_ylim(-10, 165)
axin1.hlines(100, 0, begin1, linestyles="dashed", color="#C00000", linewidth=4, label=f"Ratio = {round(1-begin1, 2)}, 100")
axin1.vlines(begin1, 0, 100, linestyles="dashed", color="#C00000", linewidth=4)
axin2.set_ylim(-0.01, 0.4)
axin.legend(loc=(0.08, 0.8), prop={"family": "Times New Roman", "size": 38})
plt.savefig(r"/root/git/Project_develop/figures/fig2_xr_frequency_delete_num_less_100.jpeg", dpi=400)


