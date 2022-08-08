from matplotlib import pyplot as plt

import prepare_dataset as pds
import learning_model as lm
import seaborn as sns


if __name__ == '__main__':

    fileNames = ['covidDatapol', 'covidDataukr']

    patients_info_filename = rf'\{fileNames[0]}.csv'

    dataframe = pds.prepare_dataset_core(patients_info_filename)

    # building ML
    col_to_predict = ['new_cases']#'new_deaths',

    sns.set()
    cols = ['new_cases', 'new_deaths', 'date']
    sns.pairplot(dataframe[cols], size=2.5)
    plt.show(block=False)
    for col in col_to_predict:
        lm.MAIN_build_and_score_ml_model_core(dataframe, col, fileNames[0])


    patients_info_filename = rf'\{fileNames[1]}.csv'
    dataframe = pds.prepare_dataset_core(patients_info_filename)
    # building ML
    sns.set()
    cols = ['new_cases', 'new_deaths', 'date']
    sns.pairplot(dataframe[cols], size=2.5)
    plt.show(block=False)
    for col in col_to_predict:
        lm.MAIN_build_and_score_ml_model_core(dataframe, col, fileNames[1])
    plt.show()
    input('Enter your input:')
