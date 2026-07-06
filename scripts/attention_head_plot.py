import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# --- DEFINICJA MODELU ---
# (Upewnij się, że używasz tej samej klasy co poprzednio)
class TinyTicTacToeGPT(nn.Module):
    def __init__(self, d_model=128, num_layers=3, nhead=8):
        super().__init__()
        self.embedding = nn.Embedding(11, d_model)
        self.pos_encoder = nn.Embedding(10, d_model)
        # UWAGA: Dodajemy batch_first=True, aby ułatwić zarządzanie danymi
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


def generuj_heatmapy_uwagi(sciezka_modelu, sekwencja_gry, glowa_docelowa=5, warstwa_docelowa=1):
    """
    Funkcja ekstrahująca i wizualizująca wagi uwagi (attention weights) z wybranej głowy.
    Domyślnie używa L1.G5 (warstwa 1, głowa 5), która okazała się najważniejsza w badaniach.
    """
    print(f"Inicjalizacja analizy dla gry: {sekwencja_gry}")

    # 1. Wczytanie wytrenowanego modelu
    device = torch.device('cpu')  # Dla bezpieczeństwa podczas ewaluacji
    model = TinyTicTacToeGPT(d_model=128, num_layers=3, nhead=8).to(device)
    model.load_state_dict(torch.load(sciezka_modelu, map_location=device, weights_only=True))
    model.eval()

    # 2. Przygotowanie wejścia (tensor 1 x Długość Gry)
    input_tensor = torch.tensor([sekwencja_gry], dtype=torch.long).to(device)
    seq_len = input_tensor.size(1)

    # Mechanizm maski (aby zachować spójność z procesem wnioskowania modelu)
    mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)

    przechwycone_wagi = {}

    # 3. Zdefiniowanie tzw. 'Hooka', by wyciągnąć macierz uwagi (zanim zostanie zwrócona do modelu)
    # Metoda MultiheadAttention w PyTorchu pozwala zwrócić wagi, musimy to zasymulować lub przechwycić
    def zbierz_uwage_hook(module, wejscie, wyjscie):
        # Wyjście z warstwy atencji to tuple: (attn_output, attn_output_weights)
        # Przy standardowym nn.TransformerEncoderLayer nie zwraca on jednak sam z siebie macierzy wag.
        # W celu jej uzyskania w PyTorchu, wchodzimy prosto do wbudowanej funkcji i "podsłuchujemy" dane wejściowe.

        q = module.in_proj_weight[:128, ...] @ wejscie[0].transpose(0, 1)
        k = module.in_proj_weight[128:256, ...] @ wejscie[0].transpose(0, 1)

        # W praktyce z PyTorcha trudniej się "wydłubuje" czystą macierz wag w gotowej warstwie TransformerEncoderLayer.
        # Bez przebudowy całego modelu najłatwiej podsłuchać funkcję zwracającą te wagi.
        pass

    print("Przygotowanie dedykowanego przejścia, aby uzyskać surowe wagi uwagi...")

    # Zamiast hooka, najskuteczniejszym sposobem bez zmiany klasy modelu w PyTorch jest "rozpakowanie" warstwy:
    warstwa_wskazana = model.transformer.layers[warstwa_docelowa]
    # Przebieg danych do momentu wejścia do atencji
    with torch.no_grad():
        positions = torch.arange(0, seq_len, device=device).unsqueeze(0)
        zbudowane_wejscie = model.embedding(input_tensor) + model.pos_encoder(positions)

        # Przechodzimy przez ewentualne poprzednie warstwy
        temp_out = zbudowane_wejscie
        for i in range(warstwa_docelowa):
            temp_out = model.transformer.layers[i](temp_out, src_mask=mask)

        # Wchodzimy do "serca" warstwy i pobieramy wagi (MultiheadAttention)
        # UWAGA: w PyTorch 2.x standardowe module.self_attn() z average_attn_weights=False by dało wagi wszystkich głów
        attn_out, attn_weights_raw = warstwa_wskazana.self_attn(
            temp_out, temp_out, temp_out, attn_mask=mask, average_attn_weights=False
        )

    # attn_weights_raw ma wymiary: [Batch, Liczba_Głów, T_docelowe, T_źródłowe]
    # np. [1, 8, seq_len, seq_len]
    wagi_docelowej_glowy = attn_weights_raw[0, glowa_docelowa].numpy()

    # 4. Tworzenie czytelnego i eleganckiego wykresu Heatmap
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")

    # Podpisy osi (jaki to był ruch)
    etykiety_ruchow = [f"Ruch {i + 1}\n(Pole {sekwencja_gry[i]})" for i in range(seq_len)]

    heatmap = sns.heatmap(
        wagi_docelowej_glowy,
        annot=True,  # Wyświetla wartości liczbowe
        fmt=".2f",  # Do 2 miejsc po przecinku
        cmap="YlGnBu",  # Kolorystyka (od jasnego żółtego do ciemnoniebieskiego)
        cbar_kws={'label': 'Siła Uwagi (Attention Weight)'},
        xticklabels=etykiety_ruchow,
        yticklabels=etykiety_ruchow
    )

    plt.title(
        f"Rozkład Uwagi (Attention Pattern) - Warstwa L{warstwa_docelowa}, Głowa G{glowa_docelowa}\nGra: {sekwencja_gry}",
        fontsize=14, fontweight="bold")
    plt.xlabel("Z jakiego ruchu (Zwracana uwaga DO)", fontsize=12)
    plt.ylabel("W którym kroku (Zwracana uwaga OD)", fontsize=12)

    plt.tight_layout()
    # Zapis
    import os
    os.makedirs('../plots/heatmaps', exist_ok=True)
    nazwa_pliku = f"../plots/heatmaps/L{warstwa_docelowa}_G{glowa_docelowa}_{''.join(map(str, sekwencja_gry))}.png"
    plt.savefig(nazwa_pliku, dpi=300)
    print(f"Zapisano wykres do: {nazwa_pliku}")
    plt.show()


if __name__ == "__main__":
    SCIEZKA_MODELU = '../models/transformer/tictactoe_model.pth'  # Ustaw odpowiednią ścieżkę
    # Analizujemy tę samą grę co w poprzednich przykładach, pełne 9 ruchów lub wygrana
    gra_testowa = [0, 1, 4, 8, 3, 5, 6]  # Wygrana X w 7 ruchach
    generuj_heatmapy_uwagi(SCIEZKA_MODELU, gra_testowa, glowa_docelowa=5, warstwa_docelowa=1)