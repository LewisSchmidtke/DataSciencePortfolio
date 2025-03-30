# Pytorch for Models and training
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.utils.data as tud

# Extract data
df = pd.read_csv('data/Stock Market Dataset.csv')
df = df.iloc[::-1] # Flip dataframe so data is chronological
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")  # Convert to datetime

# This will roughly translate to an 80/20 split for training and testing
training_df = df.iloc[:1000]
testing_df = df.iloc[1000:]

# TODO: 1. Predict the next day stock price based on the previous 10 days for the S&P500
# TODO: 2. Predict the next day stock price based on the previous 10 days stock price and volatility for indiv. stocks

class SingleVariableTimeSeriesDataset(tud.Dataset):
    def __init__(self, dataframe, data_column = "S&P_500_Price", previous_days=10,):
        self.dataframe = dataframe
        self.data = dataframe[data_column].to_numpy()
        self.window_size = previous_days
        self.convert_price_to_float()

        self.samples = []
        self.create_data_samples()

    def convert_price_to_float(self):
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

class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # Add dropout

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# Initialize both train and test dataset
train_dataset = SingleVariableTimeSeriesDataset(training_df)
test_dataset = SingleVariableTimeSeriesDataset(testing_df)

# Create both train and test dataloaders | drop_last=True : drops last data entries when smaller than input
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=False, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=True)

# Initialize the model with input=window_size
simple_nn = SimpleNN(input_size=10)
criterion = nn.MSELoss() # Use standard mean squared Error
optimizer = torch.optim.SGD(simple_nn.parameters(), lr=1e-9) # Use standard Stochastic gradient Descent

simple_nn.to(torch.float32)

# inputs and targets are good - value and shape

epochs = 200
for epoch in range(epochs):
    simple_nn.train() # Set to training mode, to allow gradient calculation
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(torch.float32)
        target = target.to(torch.float32)
        optimizer.zero_grad() # Zeros the gradients
        output = simple_nn(data) # forward pass
        loss = criterion(output.squeeze(), target) # calculate loss
        loss.backward() # back propagation
        optimizer.step() # weight and bias optimization
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

simple_nn.eval()  # Set model to evaluation mode
test_loss = 0.0
with torch.no_grad():  # Disable gradient calculation
    for data, target in test_loader:
        data = data.view(data.size(0), -1)  # Flatten input for NN
        output = simple_nn(data)
        loss = criterion(output.squeeze(), target)
        test_loss += loss.item()

avg_test_loss = test_loss / len(test_loader)
print(f'Test Loss: {avg_test_loss:.4f}')