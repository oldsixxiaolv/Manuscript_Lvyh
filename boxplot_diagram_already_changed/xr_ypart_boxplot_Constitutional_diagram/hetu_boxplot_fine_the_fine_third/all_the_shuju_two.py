# -- coding:utf-8 --
def allshuju_two(r, frequency):
    """这个函数是对r从0-1的数据进行处理，得到的是r为0的个数，r为0.01的个数以此类推到r为1的个数"""
    import numpy as np
    for i in range(0, 101):
        q = 0
        if i == 0:
            for j in r:
                if (j > 0) and (j < 0.005):
                    q += 1
        elif (i != 0) and (i != 100):
            for j in r:
                if (j >= ((2 * i - 1) / 200)) and (j < (2 * i + 1) / 200):
                    q += 1
        elif i == 100:
            for j in r:
                if (j >= 0.995) and (j < 1):
                    q += 1
        else:
            frequency.append(np.NaN)
        frequency.append(q)
    frequency.reverse()
