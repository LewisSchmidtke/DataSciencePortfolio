# TODO: 1. Predict the next day stock price based on the previous 10 days for the S&P500
# TODO: 2. Predict the next day stock price based on the previous 10 days stock price and volatility for indiv. stocks

import pandas as pd
import numpy as np
import torch
from torch import nn

import seaborn as sns
import matplotlib.pyplot as plt

from datasets import SingleVariableTimeSeriesDataset, DoubleVariableTimeSeriesDataset
from models import SimpleNN, SequenceVectorLSTM

# Extract data
df = pd.read_csv('../data/Stock Market Dataset.csv')
df = df.iloc[::-1] # Flip dataframe so data is chronological
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")  # Convert to datetime

# This will roughly translate to an 80/20 split for training and testing
training_df = df.iloc[:1000]
testing_df = df.iloc[1000:]

# df['S&P_500_Price']= df['S&P_500_Price'].astype(str).str.replace(",", "")
# df['S&P_500_Price']= df['S&P_500_Price'].astype(float)
# sns.lineplot(data=df, x="Date", y="S&P_500_Price")
# plt.show()

double_var_train = DoubleVariableTimeSeriesDataset(training_df)
double_var_test = DoubleVariableTimeSeriesDataset(testing_df)
double_var_train_loader = torch.utils.data.DataLoader(double_var_train, batch_size=64, shuffle=True)
double_var_test_loader = torch.utils.data.DataLoader(double_var_test, batch_size=1, shuffle=True)

# Initialize both train and test dataset
train_dataset = SingleVariableTimeSeriesDataset(training_df, data_column="Nasdaq_100_Price",previous_days=10)
test_dataset = SingleVariableTimeSeriesDataset(testing_df, data_column="Nasdaq_100_Price", previous_days=10)

# Create both train and test dataloaders | drop_last=True : drops last data entries when smaller than input
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)

# Set learning rates
LEARNING_RATE_NN = 1e-9
EPOCHS_NN = 1000
LEARNING_RATE_LSTM = 0.01
EPOCHS_LSTM = 1000
# Variables to check if training should be done
NN_TRAINING = False
LSTM_TRAINING = True

# Initialize Simple feedforward network.
simple_nn = SimpleNN(input_size=train_dataset.window_size)

criterion_nn = nn.MSELoss() # Mean squared Error
optimizer_nn = torch.optim.SGD(simple_nn.parameters(), lr=LEARNING_RATE_NN) # Stochastic gradient Descent
simple_nn.to(torch.float32)

# Lstm input size = 2 because price and vol are used per day
lstm = SequenceVectorLSTM(input_size=2)
criterion_lstm = nn.MSELoss()
optimizer_lstm = torch.optim.Adam(lstm.parameters(), lr=LEARNING_RATE_LSTM) # Adam optimizer
lstm.to(torch.float32)


# ----------------------- FEED-FORWARD TRAINING AND EVAL ---------------------------------------
if NN_TRAINING:
    for epoch in range(EPOCHS_NN):
        simple_nn.train() # Set to training mode, to allow gradient calculation
        running_loss = 0.0
        for _, (data, target) in enumerate(train_loader):
            data, target = data.to(torch.float32), target.to(torch.float32)
            optimizer_nn.zero_grad() # Zeros the gradients
            output = simple_nn(data).squeeze() # forward pass, .squeeze to ensure same shapes of target and output
            loss = criterion_nn(output, target) # calculate loss
            loss.backward() # back propagation
            optimizer_nn.step() # weight and bias optimization
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

    simple_nn.eval()  # Set model to evaluation mode
    test_loss = 0.0
    with torch.no_grad():  # Disable gradient calculation
        for data, target in test_loader:
            data = data.view(data.size(0), -1)  # Flatten input for NN
            output = simple_nn(data).squeeze(1)
            loss = criterion_nn(output, target)
            test_loss += loss.item()
            data_list = data.squeeze().detach().numpy().tolist()
            data_list2 = data_list
            print("target: ", target.item(), "output: ", output.item())
            data_list.append(target.item())
            data_list2.append(output.item())
            # sns.lineplot(data_list, color="b")
            # sns.lineplot(data_list2, color="r")
            # plt.show()

            sns.lineplot()

    avg_test_loss = test_loss / len(test_loader)
    print(f'Test Loss: {avg_test_loss:.4f}')
# ----------------------------------------------------------------------------------------------
# -------------------------- LSTM TRAINING AND EVAL --------------------------------------------

if LSTM_TRAINING:
    scaler = double_var_train.get_scaler()
    for epoch in range(EPOCHS_LSTM):
        lstm.train()
        running_loss = 0.0
        for inputs, targets in double_var_train_loader:
            inputs, targets = inputs.to(torch.float32), targets.to(torch.float32)

            optimizer_lstm.zero_grad()
            outputs = lstm(inputs)
            loss = criterion_lstm(outputs, targets)
            loss.backward()
            optimizer_lstm.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(double_var_train) #Calculate average loss.

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.8f}")

    with torch.no_grad():
        lstm.eval()
        for inputs, targets in double_var_test_loader:
            inputs, targets = inputs.to(torch.float32), targets.to(torch.float32)
            prediction = lstm(inputs)
            loss = criterion_lstm(prediction, targets)
            normalized_prediction = prediction.detach().numpy().reshape(-1, 1)

            # Inverse transformations
            original_target = scaler.inverse_transform(
                np.hstack((targets.numpy().reshape(-1, 1), np.zeros_like(targets.numpy().reshape(-1, 1))))
            )[:, 0]
            original_prediction = scaler.inverse_transform(
                np.hstack((normalized_prediction, np.zeros_like(normalized_prediction)))  # Keep 2D shape
            )[:, 0]  # Extract only the price column
            print("targets: ",original_target, "preds: ", original_prediction, "loss: ", loss, "\n")


