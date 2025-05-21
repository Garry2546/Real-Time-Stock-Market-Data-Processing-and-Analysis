import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from Data_preprocessing import *

class BayesianLSTM(nn.Module):
    def __init__(self, n_features, output_length, batch_size):
        super(BayesianLSTM, self).__init__()
        self.batch_size = batch_size  
        self.hidden_size_1 = 128 
        self.stacked_layers = 2 
        self.hidden_size_2 = 32  

        self.lstm1 = nn.LSTM(n_features, 
                             self.hidden_size_1, 
                             num_layers=self.stacked_layers,
                             batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_size_1,
                             self.hidden_size_2,
                             num_layers=self.stacked_layers,
                             batch_first=True)
        self.fc = nn.Linear(self.hidden_size_2, output_length)

        self.dropout_probability = 0.5

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        hidden1 = self.init_hidden1(batch_size)
        out, _ = self.lstm1(x, hidden1)

        out = F.dropout(out, p=self.dropout_probability, training=True)

        hidden2 = self.init_hidden2(batch_size)
        out, _ = self.lstm2(out, hidden2)
        out = F.dropout(out, p=self.dropout_probability, training=True)

        out = out[:, -1, :]
        y_pred = self.fc(out)
        return y_pred

    def init_hidden1(self, batch_size):

        h = torch.zeros(self.stacked_layers, batch_size, self.hidden_size_1)
        c = torch.zeros(self.stacked_layers, batch_size, self.hidden_size_1)
        return (h, c)
    
    def init_hidden2(self, batch_size):

        h = torch.zeros(self.stacked_layers, batch_size, self.hidden_size_2)
        c = torch.zeros(self.stacked_layers, batch_size, self.hidden_size_2)
        return (h, c)


n_features = X_train.shape[2]      
output_length = 1                 
batch_size = 128
n_epochs = 150
learning_rate = 0.001             

model = BayesianLSTM(n_features=n_features,
                     output_length=output_length,
                     batch_size=batch_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

X_train = np.array(X_train)
y_train = np.array(y_train)

model.train()
for epoch in range(1, n_epochs + 1):

    permutation = np.random.permutation(len(X_train))
    epoch_loss = 0
    for i in range(0, len(X_train), batch_size):
        indices = permutation[i:i + batch_size]
        batch_x = X_train[indices]
        batch_y = y_train[indices]

        X_batch = torch.tensor(batch_x, dtype=torch.float32)
        y_batch = torch.tensor(batch_y, dtype=torch.float32).unsqueeze(1) 

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    if epoch % 10 == 0:
        avg_loss = epoch_loss / ((len(X_train) // batch_size) + 1)
        print(f"Epoch {epoch}/{n_epochs}, Loss: {avg_loss:.4f}")

def predict_with_uncertainty(model, X, n_samples=100):
    model.eval()
    predictions = []
    X_tensor = torch.tensor(X, dtype=torch.float32)
  
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(X_tensor).detach().numpy()
            predictions.append(pred)
    
    predictions = np.array(predictions) 
    mean_prediction = predictions.mean(axis=0)
    prediction_std = predictions.std(axis=0)
    
    return mean_prediction, prediction_std

mean_pred, uncertainty = predict_with_uncertainty(model, X_test, n_samples=100)
print("Mean predictions (first 5):", mean_pred[:5])
print("Prediction uncertainties (std, first 5):", uncertainty[:5])

# Save the trained model.
torch.save(model.state_dict(), "/Users/Garry/Year3/Third-Year-Project/model/bayesian_lstm_model.pth")
print("Model saved as bayesian_lstm_model.pth")
