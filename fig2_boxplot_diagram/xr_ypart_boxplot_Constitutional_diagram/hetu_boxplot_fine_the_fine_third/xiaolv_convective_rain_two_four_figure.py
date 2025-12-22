# -- coding:utf-8 --
import numpy as np
import matplotlib.pyplot as plt
from boxplot_all_the_shuju_two import boxplot_allshuju_two
import pandas as pd
import seaborn as sns
from read_shuju_four_figure import read_shuju
from read_shuju_add import read_shuju_ADD
from boxplot_core_program import boxplot_core_program
from boxplot_x_labels import boxplot_x_labels
from all_the_shuju_two import allshuju_two
from x_labels import x_labels
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


# 这里是读取数据阶段
all_list = read_shuju()
# stage = duqu_excel_julei(r"C:\Users\lvyih\Desktop\stage.xlsx")
# all_list = for_(all_list, stage)
# all_list2 = read_shuju_ADD()
# all_mid = []
# for i, j in zip(all_list, all_list2):
#     all_mid.append(np.concatenate((i, j)))
# all_list = all_mid
k = 0
# 这里提醒一下：只有这里的$\mathregular{^{-1}}$才可以让我们设置的字体格式可以全部识别
name = [r'FlRate (fl' + r'$\cdot$' + r'min$\mathregular{^{-1}}$)',
        r'Fl40 (fl' + r'$\cdot$' + r'min$\mathregular{^{-1}}$' + r'$\cdot$' +
        r'100km$\mathregular{^{-2}}$)',
        r'Volume20 (10$\mathregular{^{4}}$' + r'$\cdot$' + r'km$\mathregular{^{3}}$)',
        r'Volume40 (10$\mathregular{^{3}}$' + r'$\cdot$' + r'km$\mathregular{^{3}}$)',
        r'Maxht20 (km)', r'Area20 (km$\mathregular{^{2}}$)',
        r'Maxht40 (km)', r'Area40 (km$\mathregular{^{2}}$)']
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 80
plt.rcParams["lines.color"] = "red"
##########
plt.rc('axes', linewidth=2)
plt.tick_params(width=2)
fig = plt.figure(figsize=(40, 60))
##########
# 这是对fig进行更准确地划分的东西
gs = GridSpec(150, 1, figure=fig)
legend = ["(a)", "(b)", "(c)", "(d)", "(e)"]
# average_y是我们需要求的每一个分段的flashfrequence的值。
for i in all_list[1:]:
    r = all_list[0]
    average_y = []
    labels = boxplot_x_labels()
    boxplot_allshuju_two(r, average_y, i)
    # zuizhongy是将average_y的二维数据变成一维的
    zuizhongy = []
    # rx是对应zuizhongy的数据的
    rx = []
    # labels是我们通过自定义x轴的标签
    # labels = x_labels()
    # 利用主程序把rx和zuizhongy值配对好以便我们绘画箱型图
    rx, zuizhongy = boxplot_core_program(average_y, zuizhongy, rx)
    # 将rx进行翻转，利用rx都是字符串的特性，强制将x轴翻转成左大右小
    rx.reverse()
    # 同理翻转zuizhongy与翻转的rx进行对应
    zuizhongy.reverse()
    # 创建一个DataFrame对象开始绘图
    df = pd.DataFrame({name[k]: zuizhongy, 'Ratio': rx})
    # 利用以下的代码可以对plt的图中进行x轴或者y轴标签或者刻度进行字体等的设置
    fig.add_subplot(gs[30*k:30*(k+1)])
# 利用seaborn库绘制箱型图，注意这里我们一定要把zuizhongy和rx求出来就是因为boxplot只能对一维海量数据进行箱型图绘制，注意的是这里的x轴的值可以是字符串
    # 这里的whis可以定义上限和下限距离第三（一）四分位的距离，本来这里可以用meanline这个参数，但是我们是中间空了一格必须换方法
    g = sns.boxplot(x="Ratio", y=name[k], whis=0.6, data=df, showfliers=False, color="goldenrod", width=0.3)
    # 调整x轴为自定义，利用之前弄的labels作为X轴
    g.set_xticklabels(labels, font={"family": "Times New Roman", "size": 80})
    g.tick_params(direction='out', which="major", length=10, width=2,
                  top=False, right=False)

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
    if k == 4:
        g.set_xlabel("Ratio", font={"family": "Times New Roman", "size": 80})
    else:
        g.get_xaxis().set_visible(False)
    g.yaxis.label.set_size(60)
    if k == 1 or k == 0:
        g.set_ylim(-0.1, None)
        # 这个是设置x，y轴等比例显示的
        # g.yaxis.set_major_locator(MultipleLocator(2))
    elif k == 2:
        g.set_ylim(-0.1, None)
        # 这个是设置x，y轴等比例显示的
        # g.yaxis.set_major_locator(MultipleLocator(8))
    elif k == 3:
        g.set_ylim(-0.1, None)
    elif k == 4:
        g.set_ylim(41, 58)
    else:
        g.set_ylim(7, 19)
    # 先画下平均值的点图
    """我们一定要注意的就是如果我们想要在一个图里面叠加画图，一定要保证x轴的值是一样的，否则python就会出错"""
    # 这里我们也可以注意学习一下df.groupby函数
    # h = sns.pointplot(x="r", y=name[k], data=df.groupby('r', as_index=False).mean()
                      # , scale=1, color="red", join=True)
    # 接下来要进行的是平均值点的连线，all是一个dataFrame对象我们将他的两个变量作为x和y(现在用的是median也就是中位数)
    all = df.groupby('Ratio', as_index=False).mean()
    y = all[name[k]]
    x = all["Ratio"]
    # 验证y中是空值的位置
    w1 = np.isfinite(y)
    # 连接空值左右两边的点画折线图
    g.plot(x[w1], y[w1], linestyle='-', linewidth=2.5, color="black", label="average_line", marker=".",
           markersize="10")
    g.set_xlim(-1, 33.5)
    # 隐藏y轴的坐标轴标题
    g.set(ylabel=None)
    g.yaxis.set_tick_params(pad=15)
    g.set_ylabel(name[k], fontsize=60)
    # ax_mid = g.twinx()
    # ax_mid.set_ylabel(name[k], labelpad=20, fontsize=60)
    # 这个是隐藏坐标轴的代码，坐标轴标签不影响
    # ax_mid.axes.yaxis.set_ticks([])

    # if k == 0:
    #     plt.title("Convection Parameter Boxplot of r"
    #               , font={"family": "Times New Roman", "size": 75})
    #     g.legend([f"Boxplot of r", "Average line"],
    #              loc="upper right", prop={"family": "Times New Roman", "size": 65})
    # else:
        # g.legend([f"{name[k]}boxplot_of_r", "average line"],
                 # loc="lower right", prop={"family": "Times New Roman", "size": 50})
    g.text(0.01, 0.86, legend[k], transform=g.transAxes, fontsize=70)
    k += 1
    # plt.savefig(f"C:\\Users\\lvyih\\Desktop\\PPT图\\{name[k-2]}boxplot_of_rhhh.jpeg", dpi=400)
    # plt.show()
# 这个可以自动调整子图的间距
# plt.tight_layout()
plt.savefig(f"C:\\Users\\lvyih\\Desktop\\flashrate20_maxht20_40_npixels20_40_.jpeg",
            bbox_inches="tight", dpi=400)


"""这里是可能以后要用到的代码，我暂且先放在这里"""
# plt.boxplot(average_y, labels=rrr, widths=0.5, patch_artist=True, meanline=True, showmeans=True, showfliers=False)
"""plt.boxplot(average_y, labels=rrr, widths=0.5, patch_artist=True, meanline=True, showmeans=True, showfliers=False)
plt.plot(rx, average_y1)
plt.xticks(rx, rrr)
plt.ylim(0, 55)"""
# plt.xticks(np.arange(0.01, 1.01, 0.01), rrr)


