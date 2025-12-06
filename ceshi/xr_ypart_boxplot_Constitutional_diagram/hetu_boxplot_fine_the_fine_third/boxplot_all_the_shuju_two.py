# -- coding:utf-8 --
def boxplot_allshuju_two(r, average_y, flashcount):
    """这个函数是对r从0-1的数据进行处理，得到的是r为0的个数，r为0.02的个数以此类推到r为1的个数"""
    for i in range(67, 101, 1):
        flashcount_all_count = []
        q = 0
        if i == 67:
            for j in r:
                if (j >= 0.67) and (j < 0.675):
                    flashcount_all_count.append(flashcount[q])
                q += 1
        elif (i != 67) and (i != 100):
            for j in r:
                if (j >= (0.675 + (i - 68) / 100)) and (j < 0.675 + (i - 66) / 100):
                    flashcount_all_count.append(flashcount[q])
                q += 1
        else:
            for j in r:
                if (j >= 0.995) and (j <= 1):
                    flashcount_all_count.append(flashcount[q])
                q += 1
        average_y.append(flashcount_all_count)
