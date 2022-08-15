from datetime import datetime
from matplotlib import pyplot as plt
from sktime.utils.plotting import plot_series
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_absolute_percentage_error
import core_action as ca

# 30 days
start_date = '01-25-2022'
last_date = '03-23-2022'
# 21 days
# start_date = '02-4-2022'
# last_date = '03-14-2022'

delimiter_date = '02-24-2022'
look_back = 1 #29
IS_DEBUG_MODE_ON = False


def MAIN_build_and_score_ml_model_core(df: DataFrame, col_to_predict: str, fileName: str):
    best_model = build_estimate(df, col_to_predict, fileName)
    x_train_before, y_train_before, x_test_before, y_test_before, scaler_before = prepare_data_3d(df, col_to_predict, start_date, delimiter_date)
    x_train_after, y_train_after, x_test_after, y_test_after, scaler_after = prepare_data_3d(df, col_to_predict, delimiter_date, last_date)
    # Predicting
    prediction_after = predict(best_model, scaler_after, x_test_after)
    prediction_before = predict(best_model, scaler_before, x_test_before)

    if IS_DEBUG_MODE_ON:
        plot(prediction_after, df, col_to_predict, delimiter_date, last_date)
        plot(prediction_before, df, col_to_predict, start_date, delimiter_date)
        plt.show(block=False)

    plot_final_and_estimate(prediction_before, prediction_after, df, col_to_predict, fileName)


# Plot prediction before and after war with actual data
def plot_final_and_estimate(prediction_before, prediction_after, df, col_to_predict, fileName: str):
    predict_before_war, y_valid_before, predict_after_war, y_valid_after = plot_final(prediction_before, prediction_after, df, col_to_predict)
    score_rmse_before_war, score_mape_before_war = estimate_model(predict_before_war, y_valid_before)
    score_rmse_after_war, score_mape_after_war = estimate_model(predict_after_war, y_valid_after)

    ca.writeToFile(f"Estimation before war {fileName} {col_to_predict} MAPE:" + str(score_mape_before_war))
    ca.writeToFile(f"Estimation after war {fileName} {col_to_predict} MAPE:" + str(score_mape_after_war))

    # filename = datetime.today().strftime('%Y-%m-%d') + col_to_predict + '.csv'
    # pred.to_csv(r'P:\NetRepos\CovidDataCollector\CovidAnalyzer\CsvStorage' + filename, parse_dates=[0])


def get_train_and_test_data(df, dependent_variables, delimiter_date, last_date, col_to_predict):
    cols = (['date'] + dependent_variables).append(col_to_predict)
    df = df[cols].rename(columns={'date': 'ds', 'new_cases': 'y'})

    X_tr = df.loc[(df['ds'] <= delimiter_date)]
    X_tst = df.loc[(df['ds'] >= delimiter_date)].head((datetime.strptime(last_date, "%Y-%m-%d") - datetime.strptime(delimiter_date, "%Y-%m-%d")).days)
    return X_tr, X_tst


def create_model(x_train_, y_train_, x_test_, y_test_, i):
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train_.shape[1], x_train_.shape[2])))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer="Adam", loss="mse", metrics=["mae", "acc"])

    # Train the model
    model.fit(x_train_, y_train_, batch_size=1, epochs=i, validation_data=(x_test_, y_test_))

    model.summary()
    return model


def create_dataset(dataset_creation, look_back_creation=1):
    X, Y = [], []
    for i in range(look_back_creation, len(dataset_creation)):
        a = dataset_creation[i - look_back_creation:i, 0]
        X.append(a)
        Y.append(dataset_creation[i, 0])
    return np.array(X), np.array(Y)


def plot(test_predict: np.ndarray, df: DataFrame, col_to_predict, start_index, finish_index):
    x_train, y_train, x_valid, y_valid = prepare_data_2d(df, col_to_predict, start_index, finish_index)
    plot_series(DataFrame(y_train.to_numpy().astype(float), index=x_train.index),
                DataFrame(y_valid.to_numpy().astype(float), index=y_valid.index),
                DataFrame(test_predict, index=x_valid.index), labels=["y_train", "y_test", "y_pred"])

    plt.show(block=False)


def plot_final(predict_before_war: np.ndarray, predict_after_war: np.ndarray, df: DataFrame, col_to_predict):
    x_train_before, y_train_before, x_valid_before, y_valid_before = prepare_data_2d(df, col_to_predict, start_date, delimiter_date)
    x_train_after, y_train_after, x_valid_after, y_valid_after = prepare_data_2d(df, col_to_predict, delimiter_date, last_date)
    x_train, y_train, x_valid, y_valid = prepare_data_2d(df, col_to_predict, start_date, last_date)
    plot_series(DataFrame(predict_after_war, index=x_valid_after.index),
                DataFrame(y_train_before.to_numpy().astype(float), index=x_train_before.index),
                DataFrame(predict_before_war, index=x_valid_before.index),
                DataFrame(y_valid.to_numpy().astype(float), index=y_valid.index),
                labels=["predict_after_war", "y_train", "predict_before_war", "y_test"])

    plt.show(block=False)
    return predict_before_war, y_valid_before, predict_after_war, y_valid_after


def getTrainSize(df, train_data_start, train_data_start_finish):
    # difference between dates in timedelta
    delta = datetime.strptime(train_data_start_finish, "%m-%d-%Y") - datetime.strptime(train_data_start, "%m-%d-%Y")
    finish_date_index = df.loc[df['dateReal'] == train_data_start_finish.replace("-", "/").strip('0')].first_valid_index()

    df_cut = df.loc[df['date'] < finish_date_index]
    return int(finish_date_index - delta.days), df_cut


def prepare_data_2d(df, col_to_predict: str, train_data_start, train_data_start_finish):
    train_size, varying_df = getTrainSize(df, train_data_start, train_data_start_finish)

    univariate_df = varying_df[['date', col_to_predict]].copy()
    univariate_df.columns = ['ds', 'y']

    x_train, y_train = pd.DataFrame(univariate_df.iloc[:train_size, 0]), pd.DataFrame(univariate_df.iloc[:train_size, 1])
    x_valid, y_valid = pd.DataFrame(univariate_df.iloc[train_size:, 0]), pd.DataFrame(univariate_df.iloc[train_size:, 1])
    return x_train, y_train, x_valid[:-1], y_valid[:-1]


def prepare_data_3d(df, col_to_predict: str, train_data_start, train_data_start_finish):
    train_size, varying_df = getTrainSize(df, train_data_start, train_data_start_finish)

    univariate_df = varying_df[['date', col_to_predict]].copy()
    univariate_df.columns = ['ds', 'y']

    # Convert the dataframe to a numpy array
    dataset = univariate_df.filter(['y']).values

    scaler = MinMaxScaler(feature_range=(-1, 0))
    scaled_data = scaler.fit_transform(dataset)

    # Split into train and test sets
    train, test = scaled_data[:train_size-look_back, :], scaled_data[train_size-look_back:, :]
    x_train, y_train = create_dataset(train, look_back)
    x_test, y_test = create_dataset(test, look_back)

    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
    return x_train, y_train, x_test, y_test, scaler


def build_estimate(df, col_to_predict: str, fileName: str, max_epoch_number=21):
    best_rmse = 2000
    best_mape = 0.3
    best_epoch = 0
    min_epoch = 2
    best_fit_model = None
    if IS_DEBUG_MODE_ON:
        max_epoch_number = min_epoch+1
    for i in range(min_epoch, max_epoch_number):
        x_train, y_train, x_test, y_test, scaler = prepare_data_3d(df, col_to_predict, start_date, delimiter_date)

        x_test_ = x_test
        x_train_ = x_train
        y_train_ = y_train
        y_test_ = y_test
        model = create_model(x_train_, y_train_, x_test_, y_test_, i)

        test_predict_ = predict(model, scaler, x_test)
        y_test_ = scaler.inverse_transform([y_test_])[0][:-1]  # !!! SHIFT for 1 day

        score_rmse, score_mape = estimate_model(y_test_, test_predict_)

        if best_mape > score_mape or best_rmse > score_rmse:
            best_mape = score_mape
            best_epoch = i
            best_fit_model = model
            best_rmse = score_rmse

    if IS_DEBUG_MODE_ON:
        ca.writeToFile(f"Estimation {fileName} {col_to_predict} MAPE:" + str(best_mape) + "; epoch:" + str(best_epoch))

    if best_fit_model is None:
        raise AssertionError("None of models were optimal")

    return best_fit_model


def predict(model, scaler, x_for_prediction):
    test_predict_ = model.predict(x_for_prediction)
    test_predict_ = scaler.inverse_transform(test_predict_)
    return np.delete(test_predict_, 0)  # !!! SHIFT for 1 day


def estimate_model(y_test, test_predict):
    score_rmse = np.sqrt(mean_squared_error(y_test, test_predict))
    score_mape = mean_absolute_percentage_error(y_test, test_predict)
    return score_rmse, score_mape

