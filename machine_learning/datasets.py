import numpy as np
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.utils.data as tud

class SingleVariableTimeSeriesDataset(tud.Dataset):
    def __init__(self, dataframe, data_column = "S&P_500_Price", previous_days=10):
        self.dataframe = dataframe
        self.data = dataframe[data_column].to_numpy()
        self.window_size = previous_days
        self.convert_price_to_float()

        self.samples = []
        self.create_data_samples()

    def convert_price_to_float(self):
        """
        This function converts the price data to float type. Initial data look like this: "23,987.22"
        :return: None
        """
        vector_function = np.vectorize(lambda x: float(x.replace(",","")))
        self.data = vector_function(self.data)

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        time_series, price_target = self.samples[idx]
        # Use .values to get array structure instead of Series
        return torch.tensor(time_series, dtype=torch.float32), torch.tensor(price_target, dtype=torch.float32)

    def create_data_samples(self):
        # Define constant sliding window
        left = 0
        right = self.window_size
        while right < len(self.data):
            # Append tuple consisting of the 10 entries at pos 0 and the 11th entry at pos 1 being the target
            self.samples.append((self.data[left:right], self.data[right]))
            left += 1
            right += 1

class DoubleVariableTimeSeriesDataset(tud.Dataset):
    def __init__(self, dataframe, price_column = "Nasdaq_100_Price", volume_column="Nasdaq_100_Vol.", previous_days=10):
        self.price_column = price_column
        self.data = dataframe[[price_column, volume_column]].reset_index(drop=True)
        self.window_size = previous_days
        self.convert_price_to_float()
        # normalize data
        self.scaler = MinMaxScaler()
        self.data.iloc[:, :] = self.scaler.fit_transform(self.data)
        self.samples = []
        self.create_data_samples()

    def convert_price_to_float(self):
        self.data.loc[:, self.price_column] = self.data.loc[:, self.price_column].astype(str).str.replace(",", "").astype(float)

    def get_scaler(self):
        return self.scaler

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        time_series, price_target = self.samples[idx]

        return torch.tensor(time_series, dtype=torch.float32), torch.tensor(price_target, dtype=torch.float32)

    def create_data_samples(self):
        # Define constant sliding window
        left = 0
        right = self.window_size
        while right < len(self.data):
            # Append tuple consisting of the 10 entries at pos 0 and the 11th entry at pos 1 being the target
            input_arr = self.data.iloc[left:right].to_numpy(dtype=float)
            target_price = self.data[self.price_column].iloc[right]
            self.samples.append((input_arr, target_price))
            left += 1
            right += 1