# -- coding:utf-8 --
def boxplot_x_labels():
    """这个函数用来创建0到1的间隔为0.1的列表，其中不能被10整除的地方用空格代替"""
    rrr = []
    for i in range(60, 101):
        if i % 5 == 0:
            rrr.append(i/100)
        else:
            rrr.append(" ")
    labels = list(reversed(rrr))
    return labels
