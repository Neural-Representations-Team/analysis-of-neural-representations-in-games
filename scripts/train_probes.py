import numpy as np
import torch.nn as nn
import torch
from hooks import model, przechowywane_dane

class LinearProbe(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.layer = nn.Linear(self.input_size, self.output_size)

    def forward(self, x):
        return self.layer(x)


sonda = LinearProbe(32, 27)

judge = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(sonda.parameters(), lr=0.01)

data_disk = torch.load('data/processed/dataset_testowy.pt')
aktywacje = data_disk['aktywacje']

epochs = 100

for epoch in range(epochs):
    optimizer.zero_grad()
    input_game = torch.randn(1,9)
    with torch.no_grad():
        _ = model(input_game)

    data_new = przechowywane_dane['warstwa_2']
    prediction = sonda(data_new)

    real_plane = torch.zeros(1, 27)
    loss = judge(prediction, real_plane)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")