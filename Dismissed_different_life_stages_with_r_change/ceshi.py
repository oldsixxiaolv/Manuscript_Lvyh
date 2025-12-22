# -- coding:utf-8 --
import numpy as np
import matplotlib.pyplot as plt
from boxplot_all_the_shuju_two import boxplot_allshuju_two
import pandas as pd
import seaborn as sns
from read_shuju import read_shuju
from boxplot_core_program import boxplot_core_program
from boxplot_x_labels import boxplot_x_labels
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import MultipleLocator
from matplotlib.ticker import ScalarFormatter


def round2(x):
    """由于之后要用到map函数，在此创建一个函数"""
    ok = round(x, 1)
    return ok


def researching(x, y, fre, begin=0):
    """用来找到0.9和0.8的阙值x"""
    for w in fre:
        if (w > x) and (w < y):
            jiu = w
            break
        begin += 0.01
    return begin, jiu


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


def duqu_excel_julei(path):
    import pandas as pd
    file_path = path
    df = pd.read_excel(file_path, usecols=[1], names=None)
    dali = df.values.tolist()
    result = []
    for i in dali:
        result.append(i[0])
    return result


# 这里是读取数据阶段
all_list = read_shuju()
core_samples = np.load("./core_samples.npy")
labels = np.load("./lables.npy")
print(len(all_list[0]))
print(labels)
print(len(labels))
data1 = for_(all_list, core_samples, labels, 3)
data2 = for_(all_list, core_samples, labels, 2)
data3 = for_(all_list, core_samples, labels, 1)
data4 = for_(all_list, core_samples, labels, 0)
data = for_(all_list, core_samples, labels, None, 0)
k = 0
kk = 0
# 这里提醒一下：只有这里的$\mathregular{^{-1}}$才可以让我们设置的字体格式可以全部识别
name = [
        r'Maxht20 (km)', r'R$_{eq}$20 (km)',
        r'Maxht40 (km)', r'R$_{eq}$40 (km)',
        r'FlRate (fl' + r'$\cdot$' + r'min$\mathregular{^{-1}}$)',
        r'Fl40 (fl' + r'$\cdot$' + r'(100' +
                r'km)$\mathregular{^{-2}}$' + r'$\cdot$' + r'min$\mathregular{^{-1}}$)'
        ]
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 40
plt.rcParams["lines.color"] = "red"
##########
plt.rc('axes', linewidth=2)
plt.tick_params(width=2)
fig = plt.figure(figsize=(30, 36))
##########
# 这是对fig进行更准确地划分的东西
gs = GridSpec(150, 2, figure=fig)
legend = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
# average_y是我们需要求的每一个分段的flashfrequence的值。
for i in range(0, 6):
    if k == 0 or k == 2 or k == 4:
        ax = fig.add_subplot(gs[30 * kk:30 * (kk + 1), 0])
    else:
        ax = fig.add_subplot(gs[30 * kk:30 * (kk + 1), 1])
        kk += 1
    r_Pre_MT = data4[0]
    r_MT = data[0]
    average_y_Pre_MT = []
    average_y_MT = []
    labels = boxplot_x_labels()
    boxplot_allshuju_two(r_Pre_MT, average_y_Pre_MT, data4[i+1])
    print(average_y_Pre_MT)
    boxplot_allshuju_two(r_MT, average_y_MT, data[i+1])
    # zuizhongy是将average_y的二维数据变成一维的
    r_threshold = 40
    line_width = 8
    label = [w for w in range(0, r_threshold)]
    ax.set_xticks(label)
    ax.plot(label, average_y_Pre_MT, color="g", linewidth=8, label="Pre-MT")
    # ax.fill_between(labels, average_y_Pre_MT, color="", alpha=0.6)
    ax.plot(label, average_y_MT, color="r", linewidth=8, label="MT")
    # ax.fill_between(labels, average_y_MT, color="", alpha=0.6)
    # ax.fill_between(labels, average_y_Post_MT, color="", alpha=0.6)
    ax.tick_params(axis="x", direction='out', which="major", length=10, width=2, pad=15,
                   top=False, right=False)
    ax.tick_params(axis="y", direction='out', which="major", length=15, width=2, pad=15,
                   top=False, right=False)
    xticks = ax.xaxis.get_major_ticks()
    for i in range(0, r_threshold, 5):
        xticks[i].tick1line.set_markersize(18)
    # ax.set(ylabel=None)
    ax.yaxis.set_tick_params(pad=15)
    ax.set_ylabel(name[k], fontsize=40)
    ax.set_xticklabels(labels, font={"family": "Times New Roman", "size": 40})
    ax.set_xlim(-1, r_threshold-0.2)
    # 同样的将x轴强制偏移
    # ax.get_xaxis().set_visible(False)
    # 利用以下的代码可以对plt的图中进行x轴或者y轴标签或者刻度进行字体等的设置
    if k == 0:
        ax.text(0.01, 0.89, "(a)", transform=ax.transAxes, fontsize=40)
        ax.get_xaxis().set_visible(False)
    elif k == 1:
        ax.text(0.01, 0.89, "(b)", transform=ax.transAxes, fontsize=40)
        ax.get_xaxis().set_visible(False)
    elif k == 2:
        ax.text(0.01, 0.89, "(c)", transform=ax.transAxes, fontsize=40)
        ax.get_xaxis().set_visible(False)
    elif k == 3:
        ax.text(0.01, 0.89, "(d)", transform=ax.transAxes, fontsize=40)
        ax.get_xaxis().set_visible(False)
    elif k == 4:
        ax.text(0.01, 0.89, "(e)", transform=ax.transAxes, fontsize=40)
    elif k == 5:
        ax.text(0.01, 0.89, "(e)", transform=ax.transAxes, fontsize=40)
        # 以下代码是设置x轴不显示的
        # ax.get_xaxis().set_visible(False)
    # 利用seaborn库绘制箱型图，注意这里我们一定要把zuizhongy和rx求出来就是因为boxplot只能对一维海量数据进行箱型图绘制，注意的是这里的x轴的值可以是字符串
    # 这里的whis可以定义上限和下限距离第三（一）四分位的距离，本来这里可以用meanline这个参数，但是我们是中间空了一格必须换方法
    # g = sns.boxplot(x="Ratio", y=name[k], whis=0.6, data=df, showfliers=False, color="goldenrod", width=0.3)
    # 调整x轴为自定义，利用之前弄的labels作为X轴
    # ax.set_xticklabels(labels, font={"family": "Times New Roman", "size": 80})
    # ax.tick_params(direction='out', which="major", length=10, width=2,
    #                top=False, right=False)
    # 这是设置y轴为科学计数法来计量的代码
    # y_formatter = ScalarFormatter(useMathText=True)
    # y_formatter.set_powerlimits((-2, 2))
    # g.yaxis.set_major_formatter(y_formatter)
    # g.yaxis.set_major_formatter(y_formatter)
    # g.minorticks_on()
    # g.tick_params(direction="out", which="minor", length=6, width=1.5,
    #               top=False, right=False)
    # g.set_aspect(3)
    # 这个可以设置y轴的刻度距离是多少
    # if k == 7:
    #     ax.set_xlabel("Ratio", font={"family": "Times New Roman", "size": 80})
    # else:
    #     ax.get_xaxis().set_visible(False)
    # ax.yaxis.label.set_size(60)
    # if k == 1 or k == 0:
    #     ax.set_ylim(-0.1, None)
    #     # 这个是设置x，y轴等比例显示的
    #     # g.yaxis.set_major_locator(MultipleLocator(2))
    # elif k == 2:
    #     ax.set_ylim(-0.1, None)
    #     # 这个是设置x，y轴等比例显示的
    #     # g.yaxis.set_major_locator(MultipleLocator(8))
    # elif k == 3:
    #     ax.set_ylim(-0.1, None)
    # elif k == 4:
    #     ax.set_ylim(41, 58)
    # else:
    #     ax.set_ylim(7, 19)
    # 先画下平均值的点图
    """我们一定要注意的就是如果我们想要在一个图里面叠加画图，一定要保证x轴的值是一样的，否则python就会出错"""
    # 这里我们也可以注意学习一下df.groupby函数
    # h = sns.pointplot(x="r", y=name[k], data=df.groupby('r', as_index=False).mean()
                      # , scale=1, color="red", join=True)
    # 接下来要进行的是平均值点的连线，all是一个dataFrame对象我们将他的两个变量作为x和y(现在用的是median也就是中位数)
    # all = df.groupby('Ratio', as_index=False).mean()
    # y = all[name[k]]
    # x = all["Ratio"]
    # # 验证y中是空值的位置
    # w1 = np.isfinite(y)
    # # 连接空值左右两边的点画折线图
    # ax.plot(x[w1], y[w1], linestyle='-', linewidth=2.5, color="black", label="average_line", marker=".",
    #        markersize="10")
    # ax.set_xlim(-1, 33.5)
    # # 隐藏y轴的坐标轴标题
    # ax.set(ylabel=None)
    # ax.yaxis.set_tick_params(pad=15)
    # ax.set_ylabel(name[k], fontsize=60)
    # if k == 0:
    #     ax.text(0.01, 0.86, "(a)", transform=ax.transAxes, fontsize=70)
    # elif k == 2:
    #     ax.text(0.01, 0.86, "(b)", transform=ax.transAxes, fontsize=70)
    # elif k == 3:
    #     ax.text(0.01, 0.86, "(c)", transform=ax.transAxes, fontsize=70)
    # elif k == 4:
    #     ax.text(0.01, 0.86, "(d)", transform=ax.transAxes, fontsize=70)
    # elif k == 6:
    #     ax.text(0.01, 0.86, "(e)", transform=ax.transAxes, fontsize=70)
    k += 1
plt.savefig(f"/root/git/Project_develop/figures/fig5_ceshi.jpeg",
            bbox_inches="tight", dpi=50)


"""这里是可能以后要用到的代码，我暂且先放在这里"""
# plt.boxplot(average_y, labels=rrr, widths=0.5, patch_artist=True, meanline=True, showmeans=True, showfliers=False)
"""plt.boxplot(average_y, labels=rrr, widths=0.5, patch_artist=True, meanline=True, showmeans=True, showfliers=False)
plt.plot(rx, average_y1)
plt.xticks(rx, rrr)
plt.ylim(0, 55)"""
# plt.xticks(np.arange(0.01, 1.01, 0.01), rrr)


