from matplotlib import pyplot as plt

import prepare_dataset as pds
import learning_model as lm
import seaborn as sns


if __name__ == '__main__':
    patients_info_filename = 'covidData.csv'

    dataframe = pds.prepare_dataset_core(patients_info_filename)

    # building ML
    col_to_predict = ['new_cases', 'new_deaths']
    for col in col_to_predict:
        lm.build_and_score_ml_model_core(dataframe, col)
