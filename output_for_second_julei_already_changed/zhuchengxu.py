from read_shuju import read_shuju
# from read_shuju_add import read_shuju_ADD
from pandas import DataFrame
import numpy as np


def converge(data, data_add):
    data_mid = []
    for i, j in zip(data, data_add):
        data_mid.append(np.concatenate((i, j)))
    return data_mid


all_list = read_shuju()
# all_list_add = read_shuju_ADD()
# all_list = converge(all_list, all_list_add)
df = DataFrame({"r": all_list[0], "maxht20": all_list[1],
                "maxht30": all_list[2], "maxht40": all_list[3],
                "npixels20_R": all_list[4], "npixels30_R": all_list[5], "npixels40_R": all_list[6],
                "flash20": all_list[7], "flash30": all_list[8], "flash40": all_list[9], "minir": all_list[10],
                "flashrate": all_list[11], "maxht20_minux_maxht40": all_list[12], "ellip20": all_list[13], "ellip30": all_list[14], "ellip40": all_list[15],
                "maxdbz": all_list[16], "maxht": all_list[17],
                "npx40_divide_npx20": all_list[18], "npx40_divide_npx30": all_list[19], "ellip_20_maxht20": all_list[20], "ellip_30_maxht30": all_list[21],
                "ellip_40_maxht40": all_list[22], "Volume20": all_list[23], "Volume40": all_list[24]})
df.to_excel(r"./ouput_for_julei.xlsx", sheet_name="sheet1", index=False)
