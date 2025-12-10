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


def for_(data, core_samples, labels, num=None, delete=None):
    mid = []
    if num==None:
        i_mid = data[core_samples]
        mid = i_mid[np.where(labels != delete)]
    else:
        i_mid = data[core_samples]
        mid = i_mid[np.where(labels == num)]
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


"""读取数据部分"""
r = read_shuju()
core_samples = np.load("./core_samples.npy")
labels = np.load("./lables.npy")
data1 = for_(r, core_samples, labels, 3)
data2 = for_(r, core_samples, labels, 2)
data3 = for_(r, core_samples, labels, 1)
# 这是ES的雷暴
data4 = for_(r, core_samples, labels, 0)
# 这是CS的雷暴
data = for_(r, core_samples, labels, None, 0)
# 可以由此得到r为0的个数，r为0.01的个数以此类推到r为1的个数
frequency_CS = []
frequency_ES = []
allshuju_two(data4, frequency_ES)
allshuju_two(data, frequency_CS)
# label是从1到0的对应的xlabel
label = x_labels()
label = list(map(str, label))
label[-1] = "0"
labels = []
for n in range(0, 101):
    mid = n / 100
    labels.append(float(f"{mid:.2f}"))
"""绘制图形部分"""
# 我们要注意的是如果我们要对x轴进行自定义然后又要在画图了之后进行设置，一定要用fig，ax=plt.subplots()这个函数才行
plt.rc('axes', linewidth=3)
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
ax2.plot(labels, frequency_CS, color="#003366", linewidth=8, label="Compact Storms (CS)")
ax2.fill_between(labels, frequency_CS, color="#1966B2", alpha=0.6)
ax2.plot(labels, frequency_ES, color="#FF851B", linewidth=8, label="Extensive Storms (ES)")
ax2.fill_between(labels, frequency_ES, color="#CC7326", alpha=0.6)
ax2.set_yticks(np.array(list(map(round2, list(np.arange(0, 5500, 1000))))))
ax2.set_ylabel("Frequence (#)", font={"family": "Times New Roman", "size": 75}, labelpad=20)
ax2.set_yticklabels(np.array(list(map(round2, list(np.arange(0, 5500, 1000))))),
                    font={"family": "Times New Roman", "size": 55})
ax2.set_xticks(labels)
ax2.set_xlabel("Ratio", font={"family": "Times New Roman", "size": 75})
ax2.set_xticklabels(label, font={"family": "Times New Roman", "size": 55})
# ax2.axis(labels.extend(list(map(round2, list(np.arange(0, 1.1, 0.2))))), fontsize=16)
# ax2.vlines(begin1, 0, jiu1, linestyles="dashed", color="red", label=f"r={round(1-begin1, 2)} ,  95%", linewidth=4)
# ax2.hlines(jiu1, begin1, 1, linestyles="dashed", colors="red", linewidth=4)
# ax2.vlines(1-begin2, 0, jiu2, linestyles="dashed", color="black",
#            label=f"Ratio = {round(begin2, 2)} ,  {round(jiu2 * 100)}%", linewidth=4)
# ax2.hlines(jiu2, 1-begin2, 1, linestyles="dashed", color="black", linewidth=4)
# ax2.vlines(begin3, 0, jiu3, linestyles="dashed", color="black",
#            label=f"Ratio = {round(1-begin3, 2)} ,  {round(jiu3 * 100)}%", linewidth=4)
# ax2.hlines(jiu3, begin3, 1, linestyles="dashed", color="black", linewidth=4)
# ax2.hlines(jiu1, 0, 1, linestyles="dashed", color="blue", linewidth=4)
# 指明x和y的范围
ax2.set_ylim(-120, None)
# 让r的坐标从最左边的哪里开始出现标签
ax2.set_xlim(-0.012, 1.007)
# 指明图的标签所在的位置
# ax2.legend(loc="lower right", prop={"family": "Times New Roman", "size": 63})
# ax2.set_title("TRMM tropical convective dataset\n"
#               "The distribution of different values of convective precipitation proportions",
#               font={"family": "Times New Roman", "size": 65})
# ax2.get_yaxis().set_visible(False)
ax2.tick_params(axis="x", which="major", length=7.5, width=3, pad=10)
ax2.tick_params(axis="y", which="major", length=15, width=3, pad=10)
# 手动调整一下x轴的刻度值的长度
xticks = ax2.xaxis.get_major_ticks()
for i in range(0, 101, 5):
    xticks[i].tick1line.set_markersize(15)
ax2.tick_params(axis="x", which="minor", length=8, width=2)
ax2.legend(loc=(0.62, 0.75), prop={"family": "Times New Roman", "size": 63})
# ax2.yaxis.tick_right()
plt.savefig(r"/root/git/Project_develop/figures/fig4_xr_frequency_ES_and_CS.jpeg", dpi=400)


