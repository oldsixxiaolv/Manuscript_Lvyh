import numpy as np


def color_characterized(ht40, ht20, w):
    return_one = []
    return_two = []
    return_three = []
    htt40 = list(ht40)
    htt20 = list(ht20)
    all_shuju = 0
    density_output_first_list_all = []
    print(len(set(htt40)))
    print(len(set(htt20)))
    for i in list(set(htt40)):
        for j in list(set(htt20)):
            index1 = list(np.where(np.array(htt40) == i))
            index2 = list(np.where(np.array(htt20) == j))
            index = list(set(index1[0]) & set(index2[0]))
            density_all_first = len(ht40[index])
            if density_all_first < 20:
                continue
            density_output_first = density_all_first / 102884
            density_output_first_list_all.append(density_output_first)
    density_output_first_list_all.sort()
    den_10 = density_output_first_list_all[round(len(density_output_first_list_all) * 0.1)]
    den_20 = density_output_first_list_all[round(len(density_output_first_list_all) * 0.2)]
    den_30 = density_output_first_list_all[round(len(density_output_first_list_all) * 0.3)]
    den_40 = density_output_first_list_all[round(len(density_output_first_list_all) * 0.4)]
    den_50 = density_output_first_list_all[round(len(density_output_first_list_all) * 0.5)]
    den_60 = density_output_first_list_all[round(len(density_output_first_list_all) * 0.6)]
    den_70 = density_output_first_list_all[round(len(density_output_first_list_all) * 0.7)]
    den_80 = density_output_first_list_all[round(len(density_output_first_list_all) * 0.8)]
    den_90 = density_output_first_list_all[round(len(density_output_first_list_all) * 0.9)]
    for i in list(set(htt40)):
        for j in list(set(htt20)):
            index1 = list(np.where(np.array(htt40) == i))
            index2 = list(np.where(np.array(htt20) == j))
            index = list(set(index1[0]) & set(index2[0]))
            density_all = len(ht40[index])
            if density_all < 20:
                continue
            density_output = density_all
            print(density_output)
            all_shuju += density_all
            if 0 <= density_output < 25:
                density = "grey"
            elif 25 <= density_output < 50:
                density = "deepskyblue"
            elif 50 <= density_output < 100:
                density = "green"
            elif 100 <= density_output < 200:
                density = "darkorange"
            elif 200 <= density_output < 400:
                density = "r"
            elif density_output >= 400:
                density = "purple"
            return_one.append(i)
            return_two.append(j)
            return_three.append(density)
            print(w)
            w += 1
    return return_one, return_two, return_three, all_shuju, \
        den_10*102884, den_20*102884, den_30*102884,\
        den_40*102884, den_50*102884, den_60*102884, den_70*102884,\
        den_80*102884, den_90*102884


