import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        # define layers
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU() # Set ReLu as activation function
        self.dropout = nn.Dropout(0.2)  # Add dropout for regularization

    def forward(self, x):
        # Define forward pass
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)

class SequenceVectorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(SequenceVectorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        out, _ = self.lstm(x)
        # out shape: (batch_size, sequence_length, hidden_size)
        # We only want the output from the last time step to pass into the fully connected
        out = self.fc(out[:, -1, :])
        # out shape: (batch_size, 1)
        return out.squeeze(1)
