def x_labels():
    """这个函数用来创建0到1的间隔为0.05的列表，其中不能被5整除的地方用空格代替"""
    rrr = []
    for i in range(0, 101):
        if i % 5 == 0:
            rrr.append(i/100)
        else:
            rrr.append(" ")
    labels = list(reversed(rrr))
    return labels


def x_labels_for_anix():
    rrr = []
    for i in range(0, 101):
        rrr.append(" ")
    labels = list(reversed(rrr))
    return labels
