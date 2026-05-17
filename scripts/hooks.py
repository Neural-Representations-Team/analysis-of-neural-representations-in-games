import torch
import torch.nn as nn
import os
# Dummy Model [Atrapa modelu - prosta, pusta sieć neuronowa używana wyłącznie do testowania poprawności kodu]
class DummyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Tworzymy trzy proste warstwy, żeby było do czego podpiąć haczyk
        self.warstwa_1 = nn.Linear(9, 16) # Wejście to 9 pól planszy
        self.warstwa_2 = nn.Linear(16, 32) # To będzie nasz cel do podsłuchiwania
        self.warstwa_3 = nn.Linear(32, 9) # Wyjście modelu

    def forward(self, x):
        x = self.warstwa_1(x)
        x = self.warstwa_2(x)
        x = self.warstwa_3(x)
        return x

# Uruchamiamy naszą atrapę
model = DummyTransformer()

przechowywane_dane = {}

def our_hook(modul, input, output):
    przechowywane_dane['warstwa_2'] = output.detach()

hook = model.warstwa_2.register_forward_hook(our_hook)

print("Rozpoczynam zrzut danych do pliku...")

wszystkich_aktywacji = []
wszystkie_plansze = []
liczba_gier = 100

for _ in range(liczba_gier):
    gra_testowa = torch.randn(1, 9)
    with torch.no_grad():
        _ = model(gra_testowa)

    wszystkich_aktywacji.append(przechowywane_dane['warstwa_2'].clone())
    wszystkie_plansze.append(gra_testowa)

final_activation = torch.cat(wszystkich_aktywacji, dim=0)
final_planes = torch.cat(wszystkie_plansze, dim=0)
os.makedirs('data/processed', exist_ok=True)
torch.save({
    'aktywacje': final_activation,
    'plansze': final_planes
}, 'data/processed/dataset_testowy.pt')

print("Zakończono! Dane bezpiecznie zarchiwizowane na dysku.")
