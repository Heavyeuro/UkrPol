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

start_date = '01-25-2022'
delimiter_date = '02-24-2022'
last_date = '03-23-2022'
IS_DEBUG_MODE_ON = True


def MAIN_build_and_score_ml_model_core(df: DataFrame, col_to_predict: str, fileName: str):
    best_model = build_estimate(df, col_to_predict, fileName)
    x_train, y_train, x_test, y_test, scaler = prepare_data_3d(df, col_to_predict, start_date, delimiter_date)

    # Printing training prediction results
    test_predict = predict(best_model, scaler, x_test)
    plot(test_predict, df, col_to_predict, start_date, delimiter_date)
    plt.show(block=False)
    # TODO: Add estimation "before"

    x_train, y_train, x_test, y_test, scaler = prepare_data_3d(df, col_to_predict, delimiter_date, last_date)
    real_predict = predict(best_model, scaler, x_test)
    plot(real_predict, df, col_to_predict, delimiter_date, last_date)
    # TODO: Add estimation "after"

    # TBD
    # plot_final(real_predict, df, len(df['date']), 25, col_to_predict)


def get_train_and_test_data(df, dependent_variables, delimiter_date, last_date, col_to_predict):
    cols = ['date'] + dependent_variables
    cols.append(col_to_predict)
    df = df[cols]
    df = df.rename(columns={'date': 'ds', 'new_cases': 'y'})

    mask1 = (df['ds'] <= delimiter_date)
    mask2 = (df['ds'] >= delimiter_date)
    X_tr = df.loc[mask1]
    X_tst = df.loc[mask2].head((datetime.strptime(last_date, "%Y-%m-%d")
                                - datetime.strptime(delimiter_date, "%Y-%m-%d")).days)
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


# # Making x_for_prediction
# def prepare_fh(df, col_to_predict: str):
#     fh_initial_value = len(df['date'])
#     df = df[['date', col_to_predict]]
#
#     for i in range(fh_initial_value, int(fh_initial_value*1.15)):
#         df2 = {'date': i, col_to_predict: df[col_to_predict].iloc[-1]}
#         df = df.append(df2, ignore_index=True)
#     x_train, y_train, x_test, y_test, scaler = prepare_data_3d(df, col_to_predict)
#
#     return x_test, df


def predict(model, scaler, x_for_prediction):
    test_predict_ = model.predict(x_for_prediction)
    test_predict_ = scaler.inverse_transform(test_predict_)
    return test_predict_[:, 0]


def plot(test_predict: np.ndarray, df: DataFrame, col_to_predict, start_index, finish_index):
    x_train, y_train, x_valid, y_valid = prepare_data_2d(df, col_to_predict, start_index, finish_index)
    plot_series(DataFrame(y_train.to_numpy().astype(float), index=x_train.index),
                DataFrame(y_valid.to_numpy().astype(float), index=y_valid.index),
                DataFrame(test_predict, index=x_valid.index), labels=["y_train", "y_test", "y_pred"])

    plt.show(block=False)

# Plot prediction before and after war with actual data
# def plot_final(test_predict: np.ndarray, df: DataFrame, real_length, prediction_days, col_to_predict):
#     x_train, y_train = pd.DataFrame(df.iloc[:real_length, 0]), pd.DataFrame(df.iloc[:real_length, 1])
#     x_valid, y_valid = pd.DataFrame(df.iloc[real_length:, 0]), pd.DataFrame(df.iloc[real_length:, 1])
#     x_valid = x_valid.head(prediction_days)
#     start_pred = int(fabs(len(df['date']) - real_length-len(test_predict)))
#     test_predict = test_predict[start_pred:]
#     test_predict = test_predict[:prediction_days]
#
#     index = x_valid.index
#     pred = DataFrame(data=test_predict, index=index)
#     plot_series(y_train, pred, labels=[col_to_predict, f"Predicted {prediction_days} days"])
#     plt.show()
#     filename = datetime.today().strftime('%Y-%m-%d') + col_to_predict + '.csv'
#     pred.to_csv(r'P:\NetRepos\CovidDataCollector\CovidAnalyzer\CsvStorage' + filename, parse_dates=[0])
#     print()


def getTrainSize(df, train_data_start, train_data_start_finish):
    d1 = datetime.strptime(train_data_start, "%m-%d-%Y")
    d2 = datetime.strptime(train_data_start_finish, "%m-%d-%Y")
    # difference between dates in timedelta
    delta = d2 - d1

    finish_date_index = df.loc[df['dateReal'] == train_data_start_finish.replace("-", "/").strip('0')].first_valid_index()

    df_cut = df.loc[df['date'] < finish_date_index]
    return int(finish_date_index - delta.days), df_cut


def prepare_data_2d(df, col_to_predict: str, train_data_start, train_data_start_finish):
    train_size, varying_df = getTrainSize(df, train_data_start, train_data_start_finish)

    univariate_df = varying_df[['date', col_to_predict]].copy()
    univariate_df.columns = ['ds', 'y']

    x_train, y_train = pd.DataFrame(univariate_df.iloc[:train_size, 0]), pd.DataFrame(univariate_df.iloc[:train_size, 1])
    x_valid, y_valid = pd.DataFrame(univariate_df.iloc[train_size:, 0]), pd.DataFrame(univariate_df.iloc[train_size:, 1])
    return x_train, y_train, x_valid, y_valid


def prepare_data_3d(df, col_to_predict: str, train_data_start, train_data_start_finish):
    train_size, varying_df = getTrainSize(df, train_data_start, train_data_start_finish)

    univariate_df = varying_df[['date', col_to_predict]].copy()
    univariate_df.columns = ['ds', 'y']

    data = univariate_df.filter(['y'])
    # Convert the dataframe to a numpy array
    dataset = data.values

    scaler = MinMaxScaler(feature_range=(-1, 0))
    scaled_data = scaler.fit_transform(dataset)

    # Defines the rolling window
    look_back = 1#29#52
    # Split into train and test sets
    train, test = scaled_data[:train_size-look_back, :], scaled_data[train_size-look_back:, :]

    x_train, y_train = create_dataset(train, look_back)
    x_test, y_test = create_dataset(test, look_back)

    # reshape input to be [samples, time steps, features]
    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
    return x_train, y_train, x_test, y_test, scaler


def build_estimate(df, col_to_predict: str, fileName: str, max_epoch_number=35):
    best_rmse = 2000
    best_mape = 1
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

        # Lets predict with the model
        test_predict_ = model.predict(x_test)

        # invert predictions
        test_predict_ = scaler.inverse_transform(test_predict_)
        y_test_ = scaler.inverse_transform([y_test_])

        score_rmse, score_mape = estimate_model(y_test_, test_predict_)

        if best_mape > score_mape or best_rmse > score_rmse:
            best_mape = score_mape
            best_epoch = i
            best_fit_model = model
            best_rmse = score_rmse

    # writeToFile(f"{fileName} {col_to_predict} RMSE:" + str(best_rmse))
    ca.writeToFile(f"{fileName} {col_to_predict} MAPE:" + str(best_mape) + "; epoch:" + str(best_epoch))

    return best_fit_model


def estimate_model(y_test, test_predict):
    score_rmse = np.sqrt(mean_squared_error(y_test[0], test_predict[:, 0]))
    score_mape = mean_absolute_percentage_error(y_test[0], test_predict[:, 0])
    return score_rmse, score_mape
