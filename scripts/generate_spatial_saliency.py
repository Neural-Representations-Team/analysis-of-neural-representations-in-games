import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
#TODO Naprawić te pliki i przenieść do notebooków

# Importy z naszego uporządkowanego narzędziownika
from probe_utils import przygotuj_dane, trenuj_sondy

plt.style.use('ggplot')


# --- 1. DEFINICJA MODELU GPT ---
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


def get_relative_board_state(ruchy):
    """Oblicza relatywny stan planszy dla aktualnego gracza (0: puste, 1: moje, 2: wroga)"""
    board = [0] * 9
    player_to_move = 1 if len(ruchy) % 2 == 0 else 2

    for i, move in enumerate(ruchy):
        if move > 8: continue
        player_who_made_move = 1 if i % 2 == 0 else 2
        if player_who_made_move == player_to_move:
            board[move] = 1  # Moje
        else:
            board[move] = 2  # Wroga
    return board


def generuj_przestrzenne_mapy():
    print("Szybki trening sondy do ekstrakcji Wektorów Koncepcyjnych...")
    aktywacje, relatywne_trening, _, _ = przygotuj_dane('../data/processed/dataset_pelny.pt', 1000, 200)

    # Trenujemy tylko na chwilę, aby zdobyć wagi (słownik pojęć)
    sonda_lin, _ = trenuj_sondy(aktywacje['warstwa_2'], relatywne_trening, 1000, epochs=300)
    W_probe = sonda_lin.layer.weight.data  # Kształt: [27, 128]

    print("Ładowanie umysłu modelu (Transformer)...")
    model = TinyTicTacToeGPT()
    model.load_state_dict(
        torch.load('../models/transformer/tictactoe_model.pth', map_location='cpu', weights_only=True))
    model.eval()

    # System haczyków do wyłapywania gradientów
    layer2_grad = {}

    def forward_hook(module, inp, out):
        def backward_hook(grad):
            layer2_grad['val'] = grad

        out.register_hook(backward_hook)

    model.transformer.layers[2].register_forward_hook(forward_hook)

    scenariusze = [
        ("Gotowa wygrana", [0, 3, 1, 4], 2),  # Krzyżyk decyduje się na pole 2
        ("Konieczność blokady", [0, 1, 3, 4, 8], 7)  # Krzyżyk decyduje się zablokować pole 7
    ]

    os.makedirs('../plots/spatial_saliency', exist_ok=True)

    for nazwa_scenariusza, ruchy, target_move in scenariusze:
        model.zero_grad()

        # 1. Przepuszczamy dane
        input_tensor = torch.tensor([ruchy], dtype=torch.long)
        logits = model(input_tensor)

        # 2. Wyciągamy Logit (siłę decyzyjną) dla ruchu, który model wybrał
        logit_target = logits[0, -1, target_move]

        # 3. Odpalamy propagację wsteczną, żeby zobaczyć, skąd wzięła się ta decyzja
        logit_target.backward()

        # Otrzymujemy strzałkę z warstwy drugiej w 128 wymiarach
        grad_a = layer2_grad['val'][0, -1, :]

        board_state = get_relative_board_state(ruchy)
        saliency_spatial = np.zeros(9)

        # 4. Matematyczna magjia - Iloczyn skalarny z wektorami sondy
        for i in range(9):
            s = board_state[i]
            # Odnajdujemy właściwy wiersz w macierzy sondy: [Stan * 9 + Pole]
            row_idx = s * 9 + i
            concept_vector = W_probe[row_idx]

            # Jak mocno gradient pokrywa się z tym konkretnym polem?
            score = torch.dot(grad_a, concept_vector).item()
            saliency_spatial[i] = score

        # --- WIZUALIZACJA ---
        fig, ax = plt.subplots(figsize=(6, 5))
        saliency_grid = saliency_spatial.reshape(3, 3)
        board_grid = np.array(board_state).reshape(3, 3)

        # Używamy odchylenia 'coolwarm'. Czerwony = Pole popycha do decyzji, Niebieski = Pole odrzuca od decyzji
        vmax = np.max(np.abs(saliency_grid))
        im = ax.imshow(saliency_grid, cmap='coolwarm', vmin=-vmax, vmax=vmax)
        fig.colorbar(im, label='Wpływ Przyczynowy (Gradient * Sonda)')

        for r in range(3):
            for c in range(3):
                state = board_grid[r, c]
                if state == 1:
                    ax.text(c, r, 'Moje', ha='center', va='center', fontsize=14, color='white', fontweight='bold')
                elif state == 2:
                    ax.text(c, r, 'Wróg', ha='center', va='center', fontsize=14, color='black', fontweight='bold')
                else:
                    score_val = saliency_grid[r, c]
                    ax.text(c, r, f"{score_val:.2f}\n(Puste {r * 3 + c})", ha='center', va='center', fontsize=10,
                            color='black', alpha=0.8)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Przestrzenna Istotność Pola\n{nazwa_scenariusza} (Ruch: {target_move})", fontsize=14,
                     fontweight='bold')

        nazwa_pliku = f"spatial_{nazwa_scenariusza.lower().replace(' ', '_')}.png"
        sciezka = os.path.join('../plots/spatial_saliency', nazwa_pliku)
        plt.savefig(sciezka, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Zapisano: {sciezka}")


if __name__ == '__main__':
    generuj_przestrzenne_mapy()