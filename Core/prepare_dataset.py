from pandas import DataFrame
from sklearn.impute import SimpleImputer
import core_action as ca
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Can be replaced in future by Label Encoding for categorical variable
def prepare_dataset_core(name_csv: str):
    df = ca.read_csv(name_csv)
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    missing_data.head(20)

    sid = simple_imputing_data(df, df)
    X = sid[0]
    # df = apply_encoder(sid[0])
    X['date'] = LabelEncoder().fit_transform(X['date'])
    # return X.drop(columns=['tests_units'])
    return X


# Replacing missing values (imputing) according to certain strategy
def simple_imputing_data(X_train, X_valid):
    simple_imputer = SimpleImputer(strategy='most_frequent')
    imputed_X_train = pd.DataFrame(simple_imputer.fit_transform(X_train))
    imputed_X_valid = pd.DataFrame(simple_imputer.transform(X_valid))
    # Imputation removed column names; put them back
    imputed_X_train.columns = X_train.columns
    imputed_X_valid.columns = X_valid.columns

    return imputed_X_train, imputed_X_valid


def apply_encoder(X: DataFrame):
    s = (X.dtypes == 'object')
    object_cols = list(s[s].index)

    for col in object_cols:
        X[col] = LabelEncoder().fit_transform(X[col])

    return X

