import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json

# Importy z narzędziownika
from probe_utils import przygotuj_dane, trenuj_sondy

plt.style.use('ggplot')


# --- DEFINICJA MODELU ---
class TinyTicTacToeGPT(nn.Module):
    def __init__(self, d_model=128, num_layers=3, nhead=8):
        super().__init__()
        self.embedding = nn.Embedding(11, d_model)
        self.pos_encoder = nn.Embedding(10, d_model)
        decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 11)

    def forward(self, x, mask=None):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_encoder(positions)
        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        out = self.transformer(x, mask=mask)
        return self.fc_out(out)


# --- FUNKCJA NAPRAWCZA ---
def pobierz_gry_testowe(start=1000, koniec=1200):
    """
    Pobiera oryginalne sekwencje ruchów bezpośrednio z pliku JSON,
    aby wymiary tensora idealnie pasowały do wejścia modelu (Batch x 9).
    """
    with open('../data/games.json', 'r') as plik:
        dane_gry = json.load(plik)

    sekwencje = []
    for ruchy in dane_gry[start:koniec]:
        sekwencja = ruchy[:9]
        # Dopełniamy tokenem końca gry (9), żeby każda sekwencja miała długość 9
        while len(sekwencja) < 9:
            sekwencja.append(9)
        sekwencje.append(sekwencja)

    return torch.tensor(sekwencje, dtype=torch.long)


# --- GŁÓWNA LOGIKA EKSPERYMENTU ---
def przeprowadz_ablacje():
    print("Ładowanie danych...")
    aktywacje_czyste, relatywne_trening, relatywne_test, _ = przygotuj_dane(
        sciezka_do_danych='../data/processed/dataset_pelny.pt',
        liczba_trening=1000,
        liczba_test=200
    )

    print("Krok 1: Trening sondy odniesienia (Czysty model)...")
    # Trenujemy sondę na czystej warstwie 1 (Kartograf)
    sonda_czysta, _ = trenuj_sondy(aktywacje_czyste['warstwa_1'], relatywne_trening, 1000, epochs=500)

    mysli_test_czyste = aktywacje_czyste['warstwa_1'][1000:1200].view(-1, 128)

    with torch.no_grad():
        pred_czyste = sonda_czysta(mysli_test_czyste).view(-1, 3, 9)
        wyroki_czyste = torch.argmax(pred_czyste, dim=1)
        prawdziwe_puste = (relatywne_test == 0)
        zgadniete_puste_czyste = (wyroki_czyste == 0)
        baza_fizyka_acc = (zgadniete_puste_czyste == prawdziwe_puste).float().mean().item() * 100

    print(f"Czysta skuteczność mapy przestrzennej: {baza_fizyka_acc:.1f}%\n")

    print("Krok 2: Ładowanie modelu i wstrzykiwanie haczyków do ekstrakcji w locie...")
    model = TinyTicTacToeGPT()
    model.load_state_dict(
        torch.load('../models/transformer/tictactoe_model.pth', map_location='cpu', weights_only=True))
    model.eval()

    # ZMIANA: Używamy prawidłowych, jednowymiarowych sekwencji wejściowych
    gry_testowe = pobierz_gry_testowe(1000, 1200)

    nowe_aktywacje = {}

    def ekstrakcja_hook(module, input, output):
        nowe_aktywacje['val'] = output.detach()

    model.transformer.layers[1].register_forward_hook(ekstrakcja_hook)

    wyniki_ablacji = np.zeros((3, 8))

    print("Krok 3: Rozpoczęcie operacji uszkadzania (Ablacje na wagach)...")

    d_model = 128
    nhead = 8
    head_dim = d_model // nhead

    for warstwa in range(3):
        for glowa in range(8):
            # Zapisujemy oryginalne wagi do pamięci
            oryginalne_wagi = model.transformer.layers[warstwa].self_attn.out_proj.weight.data.clone()

            start_idx = glowa * head_dim
            end_idx = start_idx + head_dim

            # FIZYCZNA ABLACJA: Zerujemy kolumny odpowiadające za wybraną głowę
            model.transformer.layers[warstwa].self_attn.out_proj.weight.data[:, start_idx:end_idx] = 0.0

            with torch.no_grad():
                # Sieć trawi poprawne dane (właściwy wymiar)
                _ = model(gry_testowe)
                mysli_uszkodzone = nowe_aktywacje['val'].view(-1, 128)

                pred_uszkodzone = sonda_czysta(mysli_uszkodzone).view(-1, 3, 9)
                wyroki_uszkodzone = torch.argmax(pred_uszkodzone, dim=1)

                zgadniete_puste = (wyroki_uszkodzone == 0)
                acc_uszkodzone = (zgadniete_puste == prawdziwe_puste).float().mean().item() * 100

            spadek = baza_fizyka_acc - acc_uszkodzone
            wyniki_ablacji[warstwa, glowa] = spadek

            # Przywracamy oryginalne wagi przed kolejną pętlą
            model.transformer.layers[warstwa].self_attn.out_proj.weight.data = oryginalne_wagi

    print("\nGenerowanie mapy obwodów...")
    os.makedirs('../plots/ablations', exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)

    sns.heatmap(wyniki_ablacji, annot=True, fmt=".1f", cmap="Reds",
                xticklabels=[f"G{i}" for i in range(8)],
                yticklabels=[f"L{i}" for i in range(3)], ax=ax,
                cbar_kws={'label': 'Spadek skuteczności mapy (%)'})

    ax.set_title("Lokalizacja Obwodu Mapy Przestrzennej (Ablacje Głów)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Głowa Uwagi (Attention Head)", fontweight='bold')
    ax.set_ylabel("Warstwa Sieci", fontweight='bold')

    sciezka = '../plots/ablations/07_obwod_mapy.png'
    plt.savefig(sciezka, dpi=300)
    plt.close(fig)
    print(f"Zapisano rygorystyczny dowód lokalizacji obwodów w: {sciezka}")


if __name__ == '__main__':
    przeprowadz_ablacje()