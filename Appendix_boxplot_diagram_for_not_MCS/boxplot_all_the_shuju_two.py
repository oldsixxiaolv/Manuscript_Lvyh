# -- coding:utf-8 --
def boxplot_allshuju_two(r, average_y, flashcount):
    """这个函数是对r从0-1的数据进行处理，得到的是r为0的个数，r为0.02的个数以此类推到r为1的个数"""
    for i in range(60, 101, 1):
        flashcount_all_count = []
        q = 0
        if i == 60:
            for j in r:
                if (j >= 0.595) and (j < 0.605):
                    flashcount_all_count.append(flashcount[q])
                q += 1
        elif (i != 60) and (i != 100):
            for j in r:
                if (j >= (0.605 + (i - 61) / 100)) and (j < 0.605 + (i - 60) / 100):
                    flashcount_all_count.append(flashcount[q])
                q += 1
        else:
            for j in r:
                if (j >= 0.995) and (j <= 1):
                    flashcount_all_count.append(flashcount[q])
                q += 1
        average_y.append(flashcount_all_count)
