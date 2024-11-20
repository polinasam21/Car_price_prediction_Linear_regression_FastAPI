import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error as MSE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
import pickle

df_train = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')


def delete_duplicates_from_train(df_train):

    df_train_without_target = df_train.drop('selling_price', axis=1)
    df_train_without_target_duplicates = df_train_without_target.duplicated(keep='first')
    df_train_1 = df_train[~df_train_without_target_duplicates]
    df_train_1 = df_train_1.reset_index()
    df_train_1 = df_train_1.drop('index', axis=1)
    
    return df_train_1


class EncoderData(BaseEstimator):

    def __init__(self):
        self.numerical_features_medians = dict()
        self.one_hot_encoding_columns = []

    def fit(self, X, y):

        X1 = X.copy(deep=True)

        X1['mileage'] = X1['mileage'].str.replace(' kmpl', '').str.replace(' km/kg', '')
        X1['mileage'] = X1['mileage'].astype(float)
        X1['engine'] = X1['engine'].str.replace(' CC', '')
        X1['engine'] = X1['engine'].astype(float)
        X1['max_power'] = X1['max_power'].str.replace(' bhp', '')
        X1['max_power'] = X1['max_power'].apply(lambda x: None if x == "" else float(x))

        # вычисление медиан числовых переменных на train
        self.numerical_features_medians['mileage'] = X1['mileage'].median()
        self.numerical_features_medians['engine'] = X1['engine'].median()
        self.numerical_features_medians['max_power'] = X1['max_power'].median()
        self.numerical_features_medians['seats'] = X1['seats'].median()

        # получение новых столбцов после one hot encoding для категориальных переменных на train
        self.one_hot_encoding_columns = pd.get_dummies(X1, columns=['fuel', 'seller_type', 'transmission', 'owner', 'seats']).columns

        return self

    def transform(self, X):

        X1 = X.copy(deep=True)

        # перевод вещественных признаков во float
        X1['mileage'] = X1['mileage'].str.replace(' kmpl', '').str.replace(' km/kg', '')
        X1['mileage'] = X1['mileage'].astype(float)
        X1['engine'] = X1['engine'].str.replace(' CC', '')
        X1['engine'] = X1['engine'].astype(float)
        X1['max_power'] = X1['max_power'].str.replace(' bhp', '')
        X1['max_power'] = X1['max_power'].apply(lambda x: None if x == "" else float(x))
        X1 = X1.drop('torque', axis=1)

        # заполнение пропусков медианами из train
        X1['mileage'] = X1['mileage'].fillna(self.numerical_features_medians['mileage'])
        X1['engine'] = X1['engine'].fillna(self.numerical_features_medians['engine'])
        X1['max_power'] = X1['max_power'].fillna(self.numerical_features_medians['max_power'])
        X1['seats'] = X1['seats'].fillna(self.numerical_features_medians['seats'])

        # перевод целочисленных значений в int
        X1['engine'] = X1['engine'].astype(int)
        X1['seats'] = X1['seats'].astype(int)

        # кодирование категориальных признаков one hot encoding
        X1 = pd.get_dummies(X1, columns=['fuel', 'seller_type', 'transmission', 'owner', 'seats'])
        X1 = X1.drop('fuel_CNG', axis=1, errors='ignore')
        X1 = X1.drop('seller_type_Dealer', axis=1, errors='ignore')
        X1 = X1.drop('transmission_Automatic', axis=1, errors='ignore')
        X1 = X1.drop('owner_First Owner', axis=1, errors='ignore')
        X1 = X1.drop('seats_9', axis=1, errors='ignore')
        X1 = X1.reindex(columns=self.one_hot_encoding_columns, fill_value=False)

        # удаление столбца name
        X1 = X1.drop('name', axis=1)

        return X1


model = Pipeline(steps=[
    ('encoder',  EncoderData()),
    ('scaler', StandardScaler()),
    ('regressor', Ridge(alpha=613.5907273413176))
])

df_train = delete_duplicates_from_train(df_train)

X_train = df_train.drop('selling_price', axis=1)
y_train = df_train['selling_price']

model.fit(X_train, y_train)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
