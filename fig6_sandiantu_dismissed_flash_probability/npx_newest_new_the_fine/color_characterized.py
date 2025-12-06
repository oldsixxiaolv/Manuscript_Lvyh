import numpy as np


def color_characterized(flash, ht30, ht20, w, hh):
    return_one = []
    return_two = []
    return_three = []
    all_shuju = 0
    htt30 = list(ht30)
    htt20 = list(ht20)
    for i in np.arange(0, 1500, 25):
        for j in np.arange(0, 2000, 25):
            index1 = list(np.where(np.array(htt30) - i < 12.5))
            index2 = list(np.where(np.array(htt20) - j < 12.5))
            index3 = list(np.where(np.array(htt30) - i > -12.5))
            index4 = list(np.where(np.array(htt20) - j > -12.5))
            index5 = list(np.where(np.array(flash) > 0))
            index = list(set(index1[0]) & set(index2[0]) & set(index3[0]) & set(index4[0]))
            indexx = list(set(index1[0]) & set(index2[0]) & set(index3[0]) & set(index4[0]) & set(index5[0]))
            density_all = len(ht30[index])
            density_flash = len(ht30[indexx])
            if density_all < 20:
                print(hh)
                hh += 1
                continue
            all_shuju += density_all
            density_output = density_flash / density_all
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


