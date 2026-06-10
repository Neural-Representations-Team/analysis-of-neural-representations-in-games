import torch
import torch.nn as nn
import os
import json


# --- DEFINICJA MODELU ---

class TinyTicTacToeGPT(nn.Module):
    # Atrapa modelu - używana wyłącznie do mapowania wag i wyciągania aktywacji
    def __init__(self, d_model=64, num_layers=2, nhead=4):
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


# --- FUNKCJE POMOCNICZE ---

def odtworz_historie_gry(sekwencja_ruchow):
    """
    Krok po kroku odtwarza fizyczny stan planszy na podstawie listy ruchów.
    Zwraca wyrównaną sekwencję (9 kroków) oraz historię plansz.
    """
    sekwencja = sekwencja_ruchow[:9]

    # Wypełnianie brakujących kroków tokenem końca gry (9)
    while len(sekwencja) < 9:
        sekwencja.append(9)

    historia_planszy = []
    aktualna_plansza = [0] * 9
    obecny_gracz = 1  # 1 = X, 2 = O

    for ruch in sekwencja:
        if ruch != 9 and ruch < 9 and aktualna_plansza[ruch] == 0:
            aktualna_plansza[ruch] = obecny_gracz
            obecny_gracz = 2 if obecny_gracz == 1 else 1

        historia_planszy.append(aktualna_plansza.copy())

    return sekwencja, historia_planszy


def ekstrahuj_aktywacje(sciezka_modelu, sciezka_json, sciezka_wyjsciowa):
    """
    Główny silnik. Ładuje model, wstrzykuje haczyki, przepuszcza gry
    i zapisuje myśli sieci na dysku.
    """
    print("Inicjalizacja modelu i wczytywanie wag...")
    model = TinyTicTacToeGPT(d_model=128, num_layers=3, nhead=8)
    model.load_state_dict(torch.load(sciezka_modelu, map_location=torch.device('cpu'), weights_only=True))
    model.eval()

    # Słownik do przechwytywania danych w locie
    przechowywane_dane = {}

    def create_hook(name):
        def hook_fn(modul, input, output):
            przechowywane_dane[name] = output.detach()

        return hook_fn

    # Wstrzykiwanie haczyków do warstw transformera
    for i in range(3):
        model.transformer.layers[i].register_forward_hook(create_hook(f'warstwa_{i}'))

    wszystkich_aktywacji = {'warstwa_0': [], 'warstwa_1': [], 'warstwa_2': []}
    wszystkie_plansze = []

    print(f"Wczytywanie gier z pliku: {sciezka_json}")
    with open(sciezka_json, 'r') as plik:
        dane_gry = json.load(plik)

    liczba_gier = len(dane_gry)
    print(f"Załadowano {liczba_gier} gier. Rozpoczynam ekstrakcję aktywacji...")

    for krok, sekwencja_ruchow in enumerate(dane_gry):

        sekwencja, historia_planszy = odtworz_historie_gry(sekwencja_ruchow)
        gra_testowa = torch.tensor([sekwencja], dtype=torch.long)

        # Przepuszczenie gry przez sieć (haczyki same zbiorą dane)
        with torch.no_grad():
            _ = model(gra_testowa)

        # Kopiowanie zebranych myśli z pamięci podręcznej do głównego archiwum
        for warstwa in ['warstwa_0', 'warstwa_1', 'warstwa_2']:
            wszystkich_aktywacji[warstwa].append(przechowywane_dane[warstwa].clone())

        wszystkie_plansze.append(torch.tensor([historia_planszy], dtype=torch.long))

        if (krok + 1) % 2000 == 0:
            print(f"Przetworzono {krok + 1} / {liczba_gier} gier...")

    # Złączenie wszystkich list w potężne, pojedyncze tensory
    print("Pakowanie danych do zapisu...")
    final_activation = {
        'warstwa_0': torch.cat(wszystkich_aktywacji['warstwa_0'], dim=0),
        'warstwa_1': torch.cat(wszystkich_aktywacji['warstwa_1'], dim=0),
        'warstwa_2': torch.cat(wszystkich_aktywacji['warstwa_2'], dim=0)
    }
    final_planes = torch.cat(wszystkie_plansze, dim=0)

    # Bezpieczny zapis na dysku
    os.makedirs(os.path.dirname(sciezka_wyjsciowa), exist_ok=True)
    torch.save({
        'aktywacje': final_activation,
        'plansze': final_planes
    }, sciezka_wyjsciowa)

    print("Zakończono! Dane bezpiecznie zarchiwizowane na dysku.")



if __name__ == "__main__":
    SCIEZKA_MODELU = '../models/transformer/tictactoe_model.pth'
    SCIEZKA_DANYCH_JSON = '../data/games.json'
    SCIEZKA_WYJSCIOWA = '../data/processed/dataset_pelny.pt'

    ekstrahuj_aktywacje(SCIEZKA_MODELU, SCIEZKA_DANYCH_JSON, SCIEZKA_WYJSCIOWA)