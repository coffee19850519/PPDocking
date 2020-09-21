import pandas as pd
import os
if __name__ == '__main__':
    csv_path = "/home/fei/Research/zdock_results/Enrichment_factor.csv"
    max_count = 1000
    save_folder = "/home/fei/Research/zdock_results/"
    prediction_redults = pd.read_csv(os.path.join(csv_path))
    dic = {}
    count_index = [str(i) for i in range(1, max_count +1)]
    for index, rows in prediction_redults.iterrows():
        complex_name = rows["complex_names"]

        for count in count_index:
            if float(rows[count])>0:
                first_non_zero = count
                break

        dic[complex_name] = first_non_zero


    success_rate_dataframe = pd.DataFrame([dic])
    # top_dataframe = pd.concat([success_rate_dataframe, top_dataframe], axis=0, sort=False).reset_index(drop=True)

    path = os.path.join(save_folder, 'first_non_zero.csv')
    success_rate_dataframe.to_csv(path, index=False)



