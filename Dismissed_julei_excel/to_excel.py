# -- coding:utf-8 --
from read_shuju import read_shuju
from pandas import DataFrame
import matplotlib.pyplot as plt
all_list = read_shuju()
df = DataFrame({"r": all_list[0], "flashrate": all_list[1], "npixels_20_R": all_list[2], "npixels_30_R": all_list[3],
                "npixels_40_R": all_list[4], "npixels40_divide_npixels20": all_list[5],
                "fls20": all_list[6], "fls30": all_list[7], "fls40": all_list[8], "maxht20": all_list[9],
                "maxht30": all_list[10], "maxht40": all_list[11], "maxht": all_list[12], "maxdbz": all_list[13]})
df.to_excel(r"C:\Users\lvyih\Desktop\x.xlsx", sheet_name="sheet1", index=False)
