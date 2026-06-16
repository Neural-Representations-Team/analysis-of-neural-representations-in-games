import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import numpy as np

# --- 0. KONFIGURACJA ŚCIEŻEK ---
SCIEZKA_MODELU = '../models/transformer/tictactoe_model.pth'
FOLDER_WYJSCIOWY = '../plots/saliency_maps'
os.makedirs(FOLDER_WYJSCIOWY, exist_ok=True)


# --- 1. DEFINICJA MODELU Z MECHANIZMEM INTERWENCJI ---
class TinyTicTacToeGPT(nn.Module):
    def __init__(self, d_model=128, num_layers=3, nhead=8):
        super().__init__()
        self.embedding = nn.Embedding(11, d_model)
        self.pos_encoder = nn.Embedding(10, d_model)
        decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 11)

    def forward(self, x, mask=None, intervene_pos=None):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)

        input_embeddings = self.embedding(x)

        # MECHANIZM INTERWENCJI (Z artykułu Othello-GPT)
        # Zmieniamy ukrytą reprezentację konkretnego pola (zerujemy ją)
        if intervene_pos is not None:
            input_embeddings[:, intervene_pos, :] = 0.0

        embeddings = input_embeddings + self.pos_encoder(positions)

        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)

        out = self.transformer(embeddings, mask=mask)
        logits = self.fc_out(out)

        return logits


# --- 2. GŁÓWNA FUNKCJA GENERUJĄCA MAPY ---
def generuj_mapy_istotnosci():
    print("Inicjalizacja modelu i wczytywanie wag...")
    model = TinyTicTacToeGPT(d_model=128, num_layers=3, nhead=8)
    # Wczytujemy wagi (upewnij się, że plik istnieje przed uruchomieniem)
    try:
        model.load_state_dict(torch.load(SCIEZKA_MODELU, map_location=torch.device('cpu'), weights_only=True))
    except FileNotFoundError:
        print(
            f"Błąd: Nie znaleziono pliku modelu w {SCIEZKA_MODELU}. Skrypt będzie kontynuował z losowymi wagami do testów.")

    model.eval()  # Tryb testowy, wyłącza niepotrzebne procesy uczenia

    scenariusze = [
        ("Gotowa wygrana (X w polu 2)", [0, 3, 1, 4]),
        ("Konieczność blokady (X w polu 7)", [0, 1, 3, 4, 8]),
        ("Początek gry (Pusta plansza)", [10]),
    ]

    print("Rozpoczynam generowanie map istotności (metoda interwencji)...")

    # Wszystko robimy bez gradientów, bo metoda opiera się na fizycznej podmianie
    with torch.no_grad():
        for nazwa_scenariusza, ruchy in scenariusze:
            print(f"  > Przetwarzanie: {nazwa_scenariusza}...")

            gra_tensor = torch.tensor([ruchy], dtype=torch.long)
            seq_len = len(ruchy)

            # KROK 1: Obliczenie bazowego prawdopodobieństwa (p0)
            logits_base = model(gra_tensor)
            # Używamy softmax, by uzyskać procentowe szanse na dany ruch
            probs_base = torch.softmax(logits_base[0, -1, :], dim=-1)

            # Sieć wybiera ruch z największą szansą
            chosen_move_idx = torch.argmax(probs_base).item()
            p0 = probs_base[chosen_move_idx].item()

            saliency_seq = np.zeros(seq_len)

            # KROK 2: Interwencja pole po polu (p_s)
            for s in range(seq_len):
                if ruchy[s] == 10:
                    continue  # Ignorujemy token początku gry

                # Przepuszczamy dane przez sieć, ale wyłączamy (zerujemy) informacje o ruchu 's'
                logits_int = model(gra_tensor, intervene_pos=s)
                probs_int = torch.softmax(logits_int[0, -1, :], dim=-1)

                # Sprawdzamy szansę DLA TEGO SAMEGO RUCHU, co na początku
                ps = probs_int[chosen_move_idx].item()

                # Ważność = bazowa szansa minus szansa po usunięciu kafelka
                saliency_seq[s] = p0 - ps

            # Normalizacja wyników do przedziału [0, 1] dla ładniejszych kolorów na mapie
            max_val = np.max(np.abs(saliency_seq))
            if max_val > 0:
                saliency_seq = np.abs(saliency_seq) / max_val
            else:
                saliency_seq = np.zeros_like(saliency_seq)

            # --- B. WIZUALIZACJA MAPY ---
            fig, ax = plt.subplots(figsize=(6, 5))

            saliency_grid = np.zeros((3, 3))
            board_state = np.zeros((3, 3), dtype=int)

            for i, move in enumerate(ruchy):
                if move == 10:
                    continue

                row, col = move // 3, move % 3
                if move < 9:
                    saliency_grid[row, col] = saliency_seq[i]
                    board_state[row, col] = 1 if i % 2 == 0 else 2

            # Mapa ciepła
            im = ax.imshow(saliency_grid, cmap='Oranges', vmin=0, vmax=1)
            cbar = fig.colorbar(im, label='Ważność pola wg modelu (Latent Saliency)')

            # Rysowanie siatki i znaków
            for r in range(3):
                for c in range(3):
                    state = board_state[r, c]
                    if state == 1:
                        ax.text(c, r, 'X', ha='center', va='center', fontsize=20, color='blue', fontweight='bold')
                    elif state == 2:
                        ax.text(c, r, 'O', ha='center', va='center', fontsize=20, color='red', fontweight='bold')
                    else:
                        ax.text(c, r, str(r * 3 + c), ha='center', va='center', fontsize=12, color='gray', alpha=0.7)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"Istotność dla decyzji:\n'{nazwa_scenariusza}'", fontsize=14, fontweight='bold')

            nazwa_pliku = f"saliency_interv_{nazwa_scenariusza.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
            sciezka_zapisu = os.path.join(FOLDER_WYJSCIOWY, nazwa_pliku)
            plt.savefig(sciezka_zapisu, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"    Zapisano mapę: {sciezka_zapisu}")

    print("Gotowe. Wszystkie mapy oparte na interwencjach zostały zapisane na dysku.")


if __name__ == '__main__':
    generuj_mapy_istotnosci()