import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
import os
import numpy as np

# --- IMPORTY Z TWOJEGO NARZĘDZIOWNIKA ---
from probe_utils import przygotuj_dane


# --- DEFINICJA MODELU I SONDY ---
class TinyTicTacToeGPT(nn.Module):
    def __init__(self, d_model=128, num_layers=3, nhead=8):
        super().__init__()
        self.embedding = nn.Embedding(11, d_model)
        self.pos_encoder = nn.Embedding(10, d_model)
        decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 11)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_encoder(positions)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        out = self.transformer(x, mask=mask)
        return self.fc_out(out)


class LinearProbe(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layer = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.layer(x)


# --- FUNKCJE POMOCNICZE ---
def get_padded_sequence(sekwencja_ruchow):
    seq = sekwencja_ruchow[:9]
    while len(seq) < 9:
        seq.append(9)
    return seq


def evaluate_physics_accuracy(probe, activations, prawdziwe_plansze):
    with torch.no_grad():
        predykcje = probe(activations).view(-1, 3, 9)
        wyroki = torch.argmax(predykcje, dim=1)
        # Fizyka: 0 (puste) vs zajęte (1 lub 2)
        skutecznosc_fizyki = ((wyroki == 0) == (prawdziwe_plansze == 0)).float().mean().item() * 100
    return skutecznosc_fizyki


def wyzeruj_glowy(model, warstwa_idx, lista_glow):
    """
    Sprzętowo odcina wybrane głowy zamykając ich wagi wyjściowe.
    """
    d_model = 128
    nhead = 8
    head_dim = d_model // nhead

    with torch.no_grad():
        out_proj = model.transformer.layers[warstwa_idx].self_attn.out_proj
        for h in lista_glow:
            start_idx = h * head_dim
            end_idx = (h + 1) * head_dim
            # Zerujemy kolumny odpowiadające za daną głowę
            out_proj.weight[:, start_idx:end_idx] = 0.0


# --- LOGIKA ABLACJI MIĘDZYWARSTWOWEJ ---
def uruchom_ablacje_miedzywarstwowa():
    print("1. Ładowanie danych i czystych aktywacji...")
    SCIEZKA_MODELU = '../models/transformer/tictactoe_model.pth'
    SCIEZKA_DANYCH_JSON = '../data/games.json'

    aktywacje, relatywne_trening, relatywne_test, _ = przygotuj_dane(
        sciezka_do_danych='../data/processed/dataset_pelny.pt',
        liczba_trening=1000,
        liczba_test=200
    )

    mysli_trening_L1 = aktywacje['warstwa_1'][:1000].view(-1, 128)

    print("2. Trening bazowej Sondy Liniowej dla Fizyki Planszy (Warstwa L1)...")
    sonda = LinearProbe(128, 27)
    optimizer = torch.optim.Adam(sonda.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1000):
        optimizer.zero_grad()
        loss = criterion(sonda(mysli_trening_L1).view(-1, 3, 9), relatywne_trening)
        loss.backward()
        optimizer.step()

    baseline_acc = evaluate_physics_accuracy(sonda, aktywacje['warstwa_1'][1000:1200].view(-1, 128), relatywne_test)
    print(f"--> Czysta skuteczność bazowa (Baseline L1): {baseline_acc:.1f}%")

    print("\n3. Przygotowywanie sekwencji testowych...")
    with open(SCIEZKA_DANYCH_JSON, 'r') as f:
        wszystkie_gry = json.load(f)

    gry_testowe = wszystkie_gry[1000:1200]
    tensory_gier = torch.tensor([get_padded_sequence(gra) for gra in gry_testowe], dtype=torch.long)

    # Funkcja do przepuszczania i pobierania L1
    def pobierz_zepsute_aktywacje(uszkodzony_model):
        zepsute_L1 = []

        def hook(mod, inp, out):
            zepsute_L1.append(out.detach())

        handle = uszkodzony_model.transformer.layers[1].register_forward_hook(hook)
        with torch.no_grad():
            uszkodzony_model(tensory_gier)
        handle.remove()
        return zepsute_L1[0].view(-1, 128)

    print("\n4. EKSPERYMENT: Podwójny Knockout (L1 + L0)...")

    # Krok A: Tylko wszystkie głowy w L1 wyłączone (zostaje 77%)
    model_tylko_l1_off = TinyTicTacToeGPT(d_model=128, num_layers=3, nhead=8)
    model_tylko_l1_off.load_state_dict(torch.load(SCIEZKA_MODELU, map_location='cpu', weights_only=True))
    model_tylko_l1_off.eval()

    wszystkie_glowy = list(range(8))  # [0, 1, 2, 3, 4, 5, 6, 7]
    wyzeruj_glowy(model_tylko_l1_off, warstwa_idx=1, lista_glow=wszystkie_glowy)
    akty_l1_off = pobierz_zepsute_aktywacje(model_tylko_l1_off)
    acc_l1_off = evaluate_physics_accuracy(sonda, akty_l1_off, relatywne_test)
    print(f" - Skuteczność po wyłączeniu wszystkich głów w TYLKO L1: {acc_l1_off:.1f}%")

    # Krok B: Wyłączamy wszystkie głowy w L1 ORAZ L0
    model_cross_layer_off = TinyTicTacToeGPT(d_model=128, num_layers=3, nhead=8)
    model_cross_layer_off.load_state_dict(torch.load(SCIEZKA_MODELU, map_location='cpu', weights_only=True))
    model_cross_layer_off.eval()

    wyzeruj_glowy(model_cross_layer_off, warstwa_idx=1, lista_glow=wszystkie_glowy)
    wyzeruj_glowy(model_cross_layer_off, warstwa_idx=0, lista_glow=wszystkie_glowy)

    akty_cross_off = pobierz_zepsute_aktywacje(model_cross_layer_off)
    acc_cross_off = evaluate_physics_accuracy(sonda, akty_cross_off, relatywne_test)
    print(f" - Skuteczność po wyłączeniu wszystkich głów w L1 ORAZ L0: {acc_cross_off:.1f}%")

    if acc_cross_off < 45.0:
        print("\nSUKCES! WYNIK POTWIERDZA HIPOTEZĘ: Warstwa L0 działała jako sprzętowe koło ratunkowe!")
    else:
        print(
            "\nHmmm, skuteczność nadal jest wysoka. To sugeruje, że to warstwa MLP lub same embeddingi mogą utrzymywać geometrię.")


if __name__ == "__main__":
    uruchom_ablacje_miedzywarstwowa()