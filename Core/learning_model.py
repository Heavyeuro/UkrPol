from math import fabs
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


def create_model(x_train_, y_train_, x_test_, y_test_, i):
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train_.shape[1], x_train_.shape[2])))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='Adam', loss='mean_squared_error')

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


def build_and_score_ml_model_core(df: DataFrame, col_to_predict: str):
    best_model = build_estimate(df, col_to_predict, 12)
    x_train, y_train, x_test, y_test, scaler = prepare_data_3d(df, col_to_predict)

    # Printing training prediction results
    test_predict = predict(best_model, scaler, x_test)
    plot(test_predict, df, col_to_predict)

    # Printing real prediction results
    fh, df_fh = prepare_fh(df, col_to_predict)
    real_predict = predict(best_model, scaler, fh)
    plot_final(real_predict, df_fh, len(df['date']), 25, col_to_predict)


# Making x_for_prediction
def prepare_fh(df, col_to_predict: str):
    fh_initial_value = len(df['date'])
    df = df[['date', col_to_predict]]

    for i in range(fh_initial_value, int(fh_initial_value*1.15)):
        df2 = {'date': i, col_to_predict: df[col_to_predict].iloc[-1]}
        df = df.append(df2, ignore_index=True)
    x_train, y_train, x_test, y_test, scaler = prepare_data_3d(df, col_to_predict)

    return x_test, df


def predict(model, scaler, x_for_prediction):
    test_predict_ = model.predict(x_for_prediction)
    test_predict_ = scaler.inverse_transform(test_predict_)
    return test_predict_[:, 0]


def plot(test_predict: np.ndarray, df: DataFrame, col_to_predict):
    x_train, y_train, x_valid, y_valid = prepare_data_2d(df, col_to_predict)
    pred = DataFrame(test_predict, index=x_valid.index)
    plot_series(y_train, y_valid, pred, labels=["y_train", "y_test", "y_pred"])
    plt.show()


def plot_final(test_predict: np.ndarray, df: DataFrame, real_length, prediction_days, col_to_predict):
    x_train, y_train = pd.DataFrame(df.iloc[:real_length, 0]), pd.DataFrame(df.iloc[:real_length, 1])
    x_valid, y_valid = pd.DataFrame(df.iloc[real_length:, 0]), pd.DataFrame(df.iloc[real_length:, 1])
    x_valid = x_valid.head(prediction_days)
    start_pred = int(fabs(len(df['date']) - real_length-len(test_predict)))
    test_predict = test_predict[start_pred:]
    test_predict = test_predict[:prediction_days]

    index = x_valid.index
    pred = DataFrame(data=test_predict, index=index)
    plot_series(y_train, pred, labels=[col_to_predict, f"Predicted {prediction_days} days"])
    plt.show()
    filename = datetime.today().strftime('%Y-%m-%d') + col_to_predict + '.csv'
    pred.to_csv(r'../../CsvStorage/' + filename, index=False)
    print()


def prepare_data_2d(df, col_to_predict: str):
    train_size = int(0.85 * len(df))

    univariate_df = df[['date', col_to_predict]].copy()
    univariate_df.columns = ['ds', 'y']

    x_train, y_train = pd.DataFrame(univariate_df.iloc[:train_size, 0]), pd.DataFrame(univariate_df.iloc[:train_size, 1])
    x_valid, y_valid = pd.DataFrame(univariate_df.iloc[train_size:, 0]), pd.DataFrame(univariate_df.iloc[train_size:, 1])
    return x_train, y_train, x_valid, y_valid


def prepare_data_3d(df, col_to_predict: str, test_data_ratio=0.85):
    train_size = int(test_data_ratio * len(df))

    univariate_df = df[['date', col_to_predict]].copy()
    univariate_df.columns = ['ds', 'y']

    data = univariate_df.filter(['y'])
    # Convert the dataframe to a numpy array
    dataset = data.values

    scaler = MinMaxScaler(feature_range=(-1, 0))
    scaled_data = scaler.fit_transform(dataset)

    # Defines the rolling window
    look_back = 52
    # Split into train and test sets
    train, test = scaled_data[:train_size-look_back,:], scaled_data[train_size-look_back:, :]

    x_train, y_train = create_dataset(train, look_back)
    x_test, y_test = create_dataset(test, look_back)

    # reshape input to be [samples, time steps, features]
    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
    return x_train, y_train, x_test, y_test, scaler


def build_estimate(df, col_to_predict: str, max_epoch_number=12,):
    best_rmse = 100000
    best_epoch = 0
    for i in range(1, max_epoch_number):
        x_train, y_train, x_test, y_test, scaler = prepare_data_3d(df, col_to_predict)

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

        # Get the root mean squared error (RMSE) and MAE
        score_rmse = np.sqrt(mean_squared_error(y_test_[0], test_predict_[:, 0]))

        if best_rmse > score_rmse:
            best_rmse = score_rmse
            best_epoch = i
    print(best_rmse)

    return create_model(x_train, y_train, x_test, y_test, best_epoch)
