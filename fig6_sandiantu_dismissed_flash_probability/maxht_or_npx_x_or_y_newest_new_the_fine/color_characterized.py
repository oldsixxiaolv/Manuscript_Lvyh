import numpy as np


def color_characterized(flash, ht30, ht20, w):
    return_one = []
    return_two = []
    return_three = []
    htt30 = list(ht30)
    htt20 = list(ht20)
    all_shuju = 0
    print(len(set(htt30)))
    print(len(set(htt20)))
    for i in list(set(htt30)):
        for j in list(set(htt20)):
            index1 = list(np.where(np.array(htt30) == i))
            index2 = list(np.where(np.array(htt20) == j))
            index3 = list(np.where(np.array(flash) > 0))
            index = list(set(index1[0]) & set(index2[0]))
            indexx = list(set(index1[0]) & set(index2[0]) & set(index3[0]))
            density_all = len(ht30[index])
            density_flash = len(ht30[indexx])
            if density_all < 20:
                continue
            density_output = density_flash / density_all
            all_shuju += density_all
            if 0 <= density_output < 0.1:
                density = "grey"
            elif 0.1 <= density_output < 0.2:
                density = "cornflowerblue"
            elif 0.2 <= density_output < 0.3:
                density = "royalblue"
            elif 0.3 <= density_output < 0.4:
                density = "deepskyblue"
            elif 0.4 <= density_output < 0.5:
                density = "darkcyan"
            elif 0.5 <= density_output < 0.6:
                density = "green"
            elif 0.6 <= density_output < 0.7:
                density = "yellow"
            elif 0.7 <= density_output < 0.8:
                density = "darkorange"
            elif 0.8 <= density_output < 0.9:
                density = "tomato"
            else:
                density = "r"
            return_one.append(i)
            return_two.append(j)
            return_three.append(density)
            print(w)
            w += 1
    return return_one, return_two, return_three, all_shuju


