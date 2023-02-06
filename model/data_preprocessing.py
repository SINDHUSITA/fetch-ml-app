import pandas as pd
import numpy as np


class TrainTestHelper():

    def __init__(self, df, transform=None):
        df.reset_index(inplace=True, drop=True)
        self.X = df.iloc[1:, -3:].values
        self.y = df.iloc[1:, 1].values

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    @staticmethod
    def data_preprocessing(dataset_path):
        daily_data = pd.read_csv(dataset_path, parse_dates=True, index_col=None)
        daily_data[['year', 'month', 'day']] = daily_data['# Date'].str.split('-', expand=True)
        daily_data[['year', 'month', 'day']] = daily_data[['year', 'month', 'day']].apply(np.float64)
        return daily_data

    # Set fixed random number seed
