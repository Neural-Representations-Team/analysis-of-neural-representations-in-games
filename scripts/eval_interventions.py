import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Importy z naszego narzędziownika
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


# Funkcja pomocnicza do odtworzenia planszy 3x3
def get_board_state(ruchy):
    board = [0] * 9
    for i, move in enumerate(ruchy):
        if move < 9:
            # Zakładamy, że X (1) zaczyna, O (2) jest drugie
            board[move] = 1 if i % 2 == 0 else 2
    return np.array(board).reshape(3, 3)


def przeprowadz_wielokrotne_interwencje():
    print("Ładowanie danych...")
    aktywacje, relatywne_trening, _, _ = przygotuj_dane('../data/processed/dataset_pelny.pt', 1000, 200)

    print("Ładowanie modelu (Transformer)...")
    model = TinyTicTacToeGPT()
    model.load_state_dict(
        torch.load('../models/transformer/tictactoe_model.pth', map_location='cpu', weights_only=True))
    model.eval()

    # --- 3 ZDEFINIOWANE SCENARIUSZE ---
    scenariusze = [
        {
            'id': 'empty_to_full_midgame',
            'title': 'Empty to Full (Mid Game)',
            'ruchy': [0, 3, 1, 4],  # X: 0, 1; O: 3, 4. Puste docelowe: 2
            'pole_docelowe': 5,
            'stan_oryginalny': 0,  # 0 = Puste
            'stan_falszywy': 2,  # 2 = Wróg
            'sila': 1.0
        },
        {
            'id': 'full_to_empty_midgame',
            'title': 'Full to Empty (Mid Game)',
            'ruchy': [0, 3, 1, 4],  # Celujemy w pole 1, które jest już zajęte przez X
            'pole_docelowe': 1,
            'stan_oryginalny': 1,  # 1 = Moje
            'stan_falszywy': 0,  # 0 = Puste
            'sila': 1.0
        },
        {
            'id': 'empty_to_full_lategame',
            'title': 'Empty to Full (Late Game Win)',
            'ruchy': [0, 1, 3, 4, 7, 8],  # X: 0, 3, 7; O: 1, 4, 8. X może wygrać na polu 6.
            'pole_docelowe': 6,
            'stan_oryginalny': 0,  # 0 = Puste
            'stan_falszywy': 2,  # 2 = Wróg
            'sila': 1.0
        }
    ]

    os.makedirs('../plots/interventions', exist_ok=True)

    # --- PĘTLA PO WARSTWACH ---
    for layer_idx in range(3):
        warstwa_nazwa = f'warstwa_{layer_idx}'
        print(f"\n{'=' * 40}\nTESTOWANIE INTERWENCJI NA WARSTWIE: {layer_idx}\n{'=' * 40}")

        print(f"Szybki trening sondy dla {warstwa_nazwa}...")
        sonda_lin, _ = trenuj_sondy(aktywacje[warstwa_nazwa], relatywne_trening, 1000, epochs=300)
        W_probe = sonda_lin.layer.weight.data

        # --- PĘTLA PO SCENARIUSZACH ---
        for scenario in scenariusze:
            print(f"  > Uruchamianie scenariusza: {scenario['title']}")

            input_tensor = torch.tensor([scenario['ruchy']], dtype=torch.long)
            pole = scenario['pole_docelowe']
            stan_oryginalny = scenario['stan_oryginalny']
            stan_falszywy = scenario['stan_falszywy']
            sila = scenario['sila']

            # Zbudowanie stanu planszy do wizualizacji
            board_3x3 = get_board_state(scenario['ruchy'])

            # --- ZMIANA: BIERZEMY 10 TOKENÓW (0-8 to plansza, 9 to koniec gry) ---
            # BIEG CZYSTY
            with torch.no_grad():
                czyste_logity = model(input_tensor)
                czyste_prawdopodobienstwa = torch.softmax(czyste_logity[0, -1, :10], dim=0).numpy()

            # BIEG Z INTERWENCJĄ
            wektor_oryginalny = W_probe[stan_oryginalny * 9 + pole]
            wektor_falszywy = W_probe[stan_falszywy * 9 + pole]

            def hook_interwencyjny(module, input, output):
                output[0, -1, :] = output[0, -1, :] - (sila * wektor_oryginalny) + (sila * wektor_falszywy)
                return output

            uchwyt_haczyka = model.transformer.layers[layer_idx].register_forward_hook(hook_interwencyjny)

            with torch.no_grad():
                oszukane_logity = model(input_tensor)
                oszukane_prawdopodobienstwa = torch.softmax(oszukane_logity[0, -1, :10], dim=0).numpy()

            uchwyt_haczyka.remove()

            # --- NOWA WIZUALIZACJA Z TOKENEM KOŃCA GRY ---
            # Plansza zajmuje tylko pierwsze 9 wartości
            clean_prob_3x3 = (czyste_prawdopodobienstwa[:9] * 100).reshape(3, 3)
            interv_prob_3x3 = (oszukane_prawdopodobienstwa[:9] * 100).reshape(3, 3)

            # Wartości dla tokenu 9
            clean_end_prob = czyste_prawdopodobienstwa[9] * 100
            interv_end_prob = oszukane_prawdopodobienstwa[9] * 100

            fig, axes = plt.subplots(1, 2, figsize=(12, 6.5), constrained_layout=True)
            fig.suptitle(f"Causal Intervention: {scenario['title']} | Layer {layer_idx}", fontsize=16,
                         fontweight='bold')

            def draw_board(ax, prob_matrix, end_prob, title, is_intervention=False):
                annot = np.empty((3, 3), dtype=object)
                mapa_stanow = {0: "Puste", 1: "X", 2: "O"}

                for r in range(3):
                    for c in range(3):
                        indeks_pola = r * 3 + c

                        if board_3x3[r, c] == 1:
                            symbol = "X"
                        elif board_3x3[r, c] == 2:
                            symbol = "O"
                        else:
                            symbol = ""

                        tekst_etykiety = symbol

                        if is_intervention and indeks_pola == pole:
                            falszywy_symbol = mapa_stanow[stan_falszywy]
                            if tekst_etykiety != "":
                                tekst_etykiety += f"\n(Zmiana na: {falszywy_symbol})"
                            else:
                                tekst_etykiety = f"(Zmiana na: {falszywy_symbol})"

                        if tekst_etykiety != "":
                            tekst_etykiety += f"\n{prob_matrix[r, c]:.1f}%"
                        else:
                            tekst_etykiety = f"{prob_matrix[r, c]:.1f}%"

                        annot[r, c] = tekst_etykiety

                sns.heatmap(prob_matrix, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=100,
                            cbar=False, ax=ax, linewidths=2, linecolor='black', square=True,
                            annot_kws={"size": 11, "weight": "bold"})

                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])

                # Dodanie wyraźnego paska/tekstu z prawdopodobieństwem końca gry pod planszą
                kolor_tekstu = "red" if end_prob > 50 else "black"
                ax.text(1.5, 3.2, f"END TOKEN (9): {end_prob:.1f}%",
                        ha='center', va='center', fontsize=14, fontweight='bold',
                        color=kolor_tekstu, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

            # Rysowanie obu plansz
            draw_board(axes[0], clean_prob_3x3, clean_end_prob, "Clean Run (Normal State)", is_intervention=False)
            draw_board(axes[1], interv_prob_3x3, interv_end_prob, f"Intervention (Modified Memory at Sq. {pole})",
                       is_intervention=True)

            sciezka = f"../plots/interventions_new/interv_{scenario['id']}_L{layer_idx}.png"
            os.makedirs(os.path.dirname(sciezka), exist_ok=True)
            plt.savefig(sciezka, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"    Zapisano wykres: {sciezka}")

if __name__ == '__main__':
    przeprowadz_wielokrotne_interwencje()