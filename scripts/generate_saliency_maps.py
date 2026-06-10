import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import json
import numpy as np
#TODO Naprawić te pliki i przenieść do notebooków

# --- 0. KONFIGURACJA ŚCIEŻEK ---
SCIEZKA_MODELU = '../models/transformer/tictactoe_model.pth'
FOLDER_WYJSCIOWY = '../plots/saliency_maps'
os.makedirs(FOLDER_WYJSCIOWY, exist_ok=True)


# --- 1. DEFINICJA MODELU (ATRAPA) ---
class TinyTicTacToeGPT(nn.Module):
    # Model musi mieć identyczną strukturę jak ten, na którym trenowaliśmy
    def __init__(self, d_model=64, num_layers=2, nhead=4):
        super().__init__()
        self.embedding = nn.Embedding(11, d_model)
        self.pos_encoder = nn.Embedding(10, d_model)
        decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 11)

    def forward(self, x, mask=None):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)

        # Wyciągamy osadzenia osobno, żeby móc obliczyć gradienty
        input_embeddings = self.embedding(x)
        embeddings = input_embeddings + self.pos_encoder(positions)

        # Generujemy maskę przyczynową (square subsequent mask)
        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)

        out = self.transformer(embeddings, mask=mask)
        logits = self.fc_out(out)

        # Zwracamy logity ORAZ osadzenia wejściowe (do gradientów)
        return logits, input_embeddings


# --- 2. GŁÓWNA FUNKCJA GENERUJĄCA MAPY ---
def generuj_mapy_istotnosci():
    print("Inicjalizacja modelu i wczytywanie wag...")
    # Model musi mieć identyczne hiperparametry
    model = TinyTicTacToeGPT(d_model=128, num_layers=3, nhead=8)
    # Wczytujemy wagi. Weights_only=True dla bezpieczeństwa.
    model.load_state_dict(torch.load(SCIEZKA_MODELU, map_location=torch.device('cpu'), weights_only=True))
    # Przełączamy model w tryb ewaluacji
    model.eval()

    # --- A. PRZYGOTOWANIE SCENARIUSZY GIER ---
    scenariusze = [
        ("Gotowa wygrana (X w polu 2)", [0, 3, 1, 4]),  # Krzyżyk gra 0, 1 -> wygrana w 2
        ("Konieczność blokady (X w polu 7)", [0, 1, 3, 4, 8]),  # Kółko ma 1, 4 -> krzyżyk musi 7
        ("Początek gry (Pusta plansza)", [10]),  # Używamy tokenu [10] jako początku
    ]

    # Mapowanie symboli dla czytelności wizualizacji
    symbole = {0: "⬜", 1: "X", 2: "O", 9: "PADDING", 10: "Początek"}

    print("Rozpoczynam generowanie map istotności...")

    for nazwa_scenariusza, ruchy in scenariusze:
        print(f"  > Przetwarzanie: {nazwa_scenariusza}...")

        # Tworzymy tensor wejściowy
        gra_tensor = torch.tensor([ruchy], dtype=torch.long)

        # Włączamy śledzenie gradientów dla osadzeń wejściowych
        # Przepuszczamy dane przez model, aby dostać logity i osadzenia
        logits, input_embeddings = model(gra_tensor)

        # input_embeddings ma kształt [batch, seq_len, d_model] -> [1, seq_len, 128]
        # Chcemy gradient względem tego tensora
        input_embeddings.retain_grad()

        # Ostatnie wyjście (dla ostatniego ruchu w sekwencji)
        target_output = logits[0, -1, :]

        # Wybieramy indeks wybranego przez model ruchu (najwyższy logit)
        chosen_move_idx = torch.argmax(target_output).item()

        # Obliczamy "score" dla wybranego ruchu. To jego gradienty nas interesują.
        score = target_output[chosen_move_idx]

        # Wsteczna propagacja: oblicz gradienty 'score' względem 'input_embeddings'
        model.zero_grad()
        score.backward()

        # Gradienty osadzeń wejściowych: [1, seq_len, 128]
        saliency_gradients = input_embeddings.grad.data.abs()

        # Agregujemy gradienty: sumujemy po wymiarze d_model, żeby dostać istotność dla każdego ruchu
        # shapes: [1, seq_len, 128] -> [1, seq_len]
        saliency_seq = saliency_gradients.sum(dim=-1).squeeze(0).numpy()

        # Normalizujemy: dzielimy przez max, żeby wartości były w zakresie [0, 1]
        if saliency_seq.max() > 0:
            saliency_seq = saliency_seq / saliency_seq.max()
        else:
            saliency_seq = np.zeros_like(saliency_seq)

        # --- B. WIZUALIZACJA MAPY (HEATMAPA NA PLANSZY) ---
        fig, ax = plt.subplots(figsize=(6, 5))

        # Przygotowujemy dane do heatmapy planszy 3x3
        # Domyślnie plansza jest pusta
        saliency_grid = np.zeros((3, 3))
        board_state = np.zeros((3, 3), dtype=int)

        # Wypełniamy planszę istotnością ruchów, które się odbyły
        for i, move in enumerate(ruchy):
            # Ignorujemy token początku [10]
            if move == 10:
                continue

            # Ruchy 0-8 odpowiadają polom 3x3
            row, col = move // 3, move % 3

            # Przypisujemy istotność i stan planszy
            if move < 9:
                saliency_grid[row, col] = saliency_seq[i]
                # Co drugi ruch to X (1), co drugi O (2)
                board_state[row, col] = 1 if i % 2 == 0 else 2

        im = ax.imshow(saliency_grid, cmap='Oranges', vmin=0, vmax=1)
        cbar = fig.colorbar(im, label='Relatywna Istotność (Gradient)')

        # Dodajemy opisy (znaki X/O lub numery pól) do siatki
        for r in range(3):
            for c in range(3):
                state = board_state[r, c]
                if state == 1:
                    ax.text(c, r, 'X', ha='center', va='center', fontsize=20, color='blue', fontweight='bold')
                elif state == 2:
                    ax.text(c, r, 'O', ha='center', va='center', fontsize=20, color='red', fontweight='bold')
                else:
                    # Jeśli pole puste, pokazujemy jego numer (0-8)
                    ax.text(c, r, str(r * 3 + c), ha='center', va='center', fontsize=12, color='gray', alpha=0.7)

        # Ustawienia osi
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Istotność dla decyzji:\n'{nazwa_scenariusza}'", fontsize=14, fontweight='bold')

        # Zapisujemy wykres na dysku
        nazwa_pliku = f"saliency_{nazwa_scenariusza.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
        sciezka_zapisu = os.path.join(FOLDER_WYJSCIOWY, nazwa_pliku)
        plt.savefig(sciezka_zapisu, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"    Zapisano mapę: {sciezka_zapisu}")

    print("Gotowe. Wszystkie mapy istotności zostały zapisane w katalogu: '../plots/saliency_maps/'.")


if __name__ == '__main__':
    generuj_mapy_istotnosci()