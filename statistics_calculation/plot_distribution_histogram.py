import pandas as pd
import matplotlib.pyplot as plt
import os
from util_function import read_txt

def plot_distribution_histgram(csv_file_name, target_name):
    # import data
    df_interface_area = pd.read_csv(csv_file_name)


    # histogram
    df_interface_area[(df_interface_area['category'] == 0) &
                      (df_interface_area['decoy_interface_area'] != 0)].decoy_interface_area.plot(kind="hist", bins=20, color="steelblue", edgecolor="black", density=True, label="negative")

    # kernel curve
    df_interface_area[(df_interface_area['category'] == 0) &
                      (df_interface_area['decoy_interface_area'] != 0)].decoy_interface_area.plot(kind="kde", color="red", label="negative kernel")

    # histogram
    df_interface_area[(df_interface_area['category'] == 1)&
                      (df_interface_area['decoy_interface_area'] != 0)].decoy_interface_area.plot(kind="hist", bins=20, color="white",
                                                                               edgecolor="black", density=True,
                                                                               label="positive")
    # kernel curve
    df_interface_area[(df_interface_area['category'] == 1)&
                      (df_interface_area['decoy_interface_area'] != 0)].decoy_interface_area.plot(kind="kde", color="green", label="positive kernel")

    plt.xlabel("interface_area")
    plt.ylabel("sample frequency")

    plt.title(target_name + " interface distribution")

    plt.legend()

    plt.show()


if __name__ == '__main__':
    folder = r'/home/fei/Research/Dataset/zdock_decoy/2_decoys_bm4_zd3.0.2_irad/5_decoy_name/'
    target_list = read_txt(os.path.join(folder, 'caseID.lst'))
    for target_name in target_list:
        try:
            plot_distribution_histgram(os.path.join(folder, target_name + '_decoy_asa.csv'), target_name)
        except:
            print('%s cannot plot'.format(target_name))
    # plot_distribution_histgram(os.path.join(folder, '1GL1_decoy_asa.csv'), '1GL1')