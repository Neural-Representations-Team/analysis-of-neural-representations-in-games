import torch
import torch.nn as nn
import os
import json
# Dummy Model [Atrapa modelu - prosta, pusta sieć neuronowa używana wyłącznie do testowania poprawności kodu]
class TinyTicTacToeGPT(nn.Module):
    # We added d_model, num_layers, and nhead as arguments here
    def __init__(self, d_model=64, num_layers=2, nhead=4):
        super().__init__()

        # 1. Update Embeddings to use d_model
        self.embedding = nn.Embedding(11, d_model)
        self.pos_encoder = nn.Embedding(10, d_model)

        # 2. Update Transformer Layer to use d_model and nhead
        decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)

        # 3. Update Transformer Encoder to use num_layers
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

        # 4. Update the final output layer to take d_model as input
        self.fc_out = nn.Linear(d_model, 11)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_encoder(positions)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        out = self.transformer(x, mask=mask)
        return self.fc_out(out)

# Uruchamiamy naszą atrapę
model = TinyTicTacToeGPT(d_model=128, num_layers=3, nhead=8)
model.load_state_dict(torch.load('../model/tictactoe_model.pth', map_location=torch.device('cpu'), weights_only=True))
model.eval()

przechowywane_dane = {}

def create_hook(name):
    def hook_fn(modul, input, output):
        przechowywane_dane[name] = output.detach()
    return hook_fn

for i in range(3):
    model.transformer.layers[i].register_forward_hook(create_hook(f'warstwa_{i}'))

print("Rozpoczynam zrzut danych do pliku...")

wszystkich_aktywacji = {
    'warstwa_0': [],
    'warstwa_1': [],
    'warstwa_2': []
}
wszystkie_plansze = []

print("Wczytuję prawdziwe dane z pliku JSON...")
sciezka_json = os.path.join('..', 'data', 'games.json')
with open(sciezka_json, 'r') as plik:
    dane_gry = json.load(plik)

liczba_gier = len(dane_gry)
print(f"Załadowano {liczba_gier} gier. Rozpoczynam zrzut klatka po klatce...")

for krok, sekwencja_ruchow in enumerate(dane_gry):

    sekwencja = sekwencja_ruchow[:9]

    while len(sekwencja) < 9:
        sekwencja.append(9)

    gra_testowa = torch.tensor([sekwencja], dtype=torch.long)

    historia_planszy = []
    aktualna_plansza = [0]*9
    obecny_gracz = 1 # 1 = X, 2= O

    for ruch in sekwencja:
        if ruch != 9 and ruch < 9 and aktualna_plansza[ruch] == 0:
            aktualna_plansza[ruch] = obecny_gracz
            obecny_gracz = 2 if obecny_gracz == 1 else 1

        historia_planszy.append(aktualna_plansza.copy())

    with torch.no_grad():
        _ = model(gra_testowa)

    for warstwa in ['warstwa_0', 'warstwa_1', 'warstwa_2']:
        wszystkich_aktywacji[warstwa].append(przechowywane_dane[warstwa].clone())

    wszystkie_plansze.append(torch.tensor([historia_planszy], dtype=torch.long))

    if (krok + 1) % 2000 == 0:
        print(f"Przetworzono {krok + 1} / {liczba_gier} gier...")

final_activation = {
    'warstwa_0': torch.cat(wszystkich_aktywacji['warstwa_0'], dim=0),
    'warstwa_1': torch.cat(wszystkich_aktywacji['warstwa_1'], dim=0),
    'warstwa_2': torch.cat(wszystkich_aktywacji['warstwa_2'], dim=0)
}
final_planes = torch.cat(wszystkie_plansze, dim=0)

os.makedirs('data/processed', exist_ok=True)

torch.save({
    'aktywacje': final_activation,
    'plansze': final_planes
}, 'data/processed/dataset_pelny.pt')

print("Zakończono! Dane bezpiecznie zarchiwizowane na dysku.")
