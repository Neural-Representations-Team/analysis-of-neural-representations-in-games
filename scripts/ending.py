import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# --- 1. DEFINICJA MODELU (Bez zmian) ---
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


# --- 2. GŁÓWNA FUNKCJA BADAWCZA (NOWA, RĘCZNA METODA) ---
def zbadaj_obwod_sedziego(model, sekwencja_gry, warstwa_docelowa, glowa_docelowa):
    """
    Funkcja "rozpakowuje" proces wnioskowania sieci krok po kroku.
    """
    model.eval()

    # 1. Tworzymy wejście
    x = torch.tensor([sekwencja_gry], dtype=torch.long)
    seq_len = x.size(1)

    with torch.no_grad():
        # 2. Przechodzimy przez warstwę osadzeń (Embeddings)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        ukryte_reprezentacje = model.embedding(x) + model.pos_encoder(positions)

        # 3. Tworzymy maskę przyczynową (Causal Mask) blokującą patrzenie w przyszłość
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)

        wagi_docelowe = None

        # 4. Ręcznie przepuszczamy danezez kolejne warstwy Transformera
        for indeks_warstwy, warstwa in enumerate(model.transformer.layers):
            if indeks_warstwy == warstwa_docelowa:
                # JESTEŚMY W WARSTWIE SĘDZIEGO!
                # Wywołujemy sam moduł atencji z argumentami wymuszającymi pełne statystyki
                _, wagi_atencji = warstwa.self_attn(
                    ukryte_reprezentacje,  # Query
                    ukryte_reprezentacje,  # Key
                    ukryte_reprezentacje,  # Value
                    attn_mask=mask,
                    need_weights=True,
                    average_attn_weights=False  # Blokada uśredniania!
                )
                wagi_docelowe = wagi_atencji
                break  # Mamy co chcieliśmy, nie musimy liczyć reszty
            else:
                # Jeśli to wcześniejsza warstwa, przechodzimy przez nią normalnie i idziemy dalej
                ukryte_reprezentacje = warstwa(ukryte_reprezentacje, src_mask=mask)

    # --- 3. WIZUALIZACJA ---
    # wagi_docelowe mają kształt: [batch_size, num_heads, seq_len, seq_len]
    wagi = wagi_docelowe[0]  # Wyciągamy nasz pojedynczy przypadek z batcha
    wagi_glowy = wagi[glowa_docelowa].cpu().numpy()  # [seq_len, seq_len]

    # Interesuje nas uwaga ostatniego tokenu w sekwencji
    uwaga_ostatniego_tokenu = wagi_glowy[-1, :]

    plt.figure(figsize=(10, 4))

    # Etykiety osi X (historia ruchów)
    etykiety_x = [f"Ruch {i + 1}\n(Pole {p})" for i, p in enumerate(sekwencja_gry)]

    # Rysujemy 1D heatmapę
    sns.heatmap(
        [uwaga_ostatniego_tokenu],
        annot=True,
        cmap="Reds",
        xticklabels=etykiety_x,
        yticklabels=["Uwaga\nprzed końcem"],
        cbar_kws={'label': 'Siła Uwagi (Attention Weight)'},
        fmt=".3f"
    )

    plt.title(f"Dowód na Obwód Sędziego: Atencja Warstwy L{warstwa_docelowa}, Głowy H{glowa_docelowa}\n"
              f"Rozkład uwagi w momencie ewaluacji wygranej", pad=15)
    plt.xlabel("Liniowa historia ruchów (sekwencja wejściowa)")
    plt.tight_layout()
    plt.savefig(f"sedzia_W{warstwa_docelowa}_G{glowa_docelowa}.png", dpi=300)
    print(f"Pomyślnie wygenerowano plik: sedzia_W{warstwa_docelowa}_G{glowa_docelowa}.png")
    plt.show()


# --- 4. URUCHOMIENIE ---
if __name__ == "__main__":
    print("Inicjalizacja modelu...")
    model = TinyTicTacToeGPT()

    # Ładowanie z wyłączonym GPU (aby zapobiec ewentualnym błędom deserializacji)
    model.load_state_dict(torch.load('../models/transformer/tictactoe_model.pth', map_location=torch.device('cpu')))

    # Symulacja wygranej X: X na 0, O na 1, X na 4, O na 2, X na 8.
    sekwencja_testowa = [0, 1, 4, 2, 8]
    warstwa = 2
    glowa = 5

    print(f"Generowanie mapy atencji dla Warstwy {warstwa}, Głowy {glowa}...")
    zbadaj_obwod_sedziego(model, sekwencja_testowa, warstwa_docelowa=warstwa, glowa_docelowa=glowa)