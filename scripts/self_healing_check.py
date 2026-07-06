import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# --- IMPORTY Z TWOJEGO NARZĘDZIOWNIKA ---
from probe_utils import przygotuj_dane, trenuj_sondy

plt.style.use('ggplot')


# --- 1. ZDEFINIUJ MODEL ---
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
        return self.transformer(x, mask=mask)


def wytrenuj_i_pobierz_wektor_konceptualny(badane_pole, warstwa='warstwa_1', d_model=128):
    print(f"\n--- Trening sondy w locie dla pola {badane_pole} na {warstwa} ---")

    aktywacje, relatywne_trening, _, _ = przygotuj_dane(
        sciezka_do_danych='../data/processed/dataset_pelny.pt',
        liczba_trening=500,
        liczba_test=10
    )

    sonda_lin, _ = trenuj_sondy(
        aktywacje_warstwy=aktywacje[warstwa],
        relatywne_trening=relatywne_trening,
        liczba_trening=500,
        epochs=300
    )

    wagi = next(sonda_lin.parameters()).detach().cpu()

    # POPRAWIONA MATEMATYKA INDEKSÓW (Zgodna z .view(-1, 3, 9))
    idx_puste = 0 * 9 + badane_pole
    idx_X = 1 * 9 + badane_pole
    idx_O = 2 * 9 + badane_pole

    waga_puste = wagi[idx_puste, :]
    waga_X = wagi[idx_X, :]
    waga_O = wagi[idx_O, :]

    # Wektor "Zajętości" (Occupancy): Pchamy w stronę X i O, odpychamy od Puste
    wektor_konceptu = ((waga_X + waga_O) / 2.0) - waga_puste

    # Normalizacja wektora
    wektor_konceptu = wektor_konceptu / torch.norm(wektor_konceptu)
    return wektor_konceptu


def badaj_strumien_residualny(sciezka_modelu, sekwencja_gry, badane_pole):
    device = torch.device('cpu')

    wektor_konceptu = wytrenuj_i_pobierz_wektor_konceptualny(badane_pole, warstwa='warstwa_1')

    print("\n--- Analiza Strumienia Residualnego ---")
    model = TinyTicTacToeGPT(d_model=128, num_layers=3, nhead=8).to(device)
    model.load_state_dict(torch.load(sciezka_modelu, map_location=device, weights_only=True))
    model.eval()

    x = torch.tensor([sekwencja_gry], dtype=torch.long).to(device)

    historia_strumienia = []
    uchwyty_hookow = []

    def pre_hook_l0(module, wejscie):
        strumien = wejscie[0][0, -1, :]
        projekcja = torch.dot(strumien, wektor_konceptu).item()
        historia_strumienia.append(("Start (Tylko Embeddings)", projekcja))

    def hook_po_warstwie(nazwa_warstwy):
        def hook(module, wejscie, wyjscie):
            strumien = wyjscie[0, -1, :]
            projekcja = torch.dot(strumien, wektor_konceptu).item()
            historia_strumienia.append((f"Po warstwie {nazwa_warstwy}", projekcja))

        return hook

    uchwyty_hookow.append(model.transformer.layers[0].register_forward_pre_hook(pre_hook_l0))
    uchwyty_hookow.append(model.transformer.layers[0].register_forward_hook(hook_po_warstwie("L0")))
    uchwyty_hookow.append(model.transformer.layers[1].register_forward_hook(hook_po_warstwie("L1")))
    uchwyty_hookow.append(model.transformer.layers[2].register_forward_hook(hook_po_warstwie("L2 (Finał)")))

    with torch.no_grad():
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=device).unsqueeze(0)
        wejscie_emb = model.embedding(x) + model.pos_encoder(positions)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
        _ = model.transformer(wejscie_emb, mask=mask)

    for uchwyt in uchwyty_hookow:
        uchwyt.remove()

    etapy = [val[0] for val in historia_strumienia]
    wartosci = [val[1] for val in historia_strumienia]

    plt.figure(figsize=(10, 6))
    # Zielony kolor bo tym razem wynik będzie pozytywny!
    plt.plot(etapy, wartosci, marker='o', linestyle='-', linewidth=3, markersize=10, color='#2ecc71')
    plt.fill_between(etapy, wartosci, color='#2ecc71', alpha=0.1)

    plt.title(
        f"Kumulacja Reprezentacji Zajętości Pola w Strumieniu Residualnym\n(Badane pole: {badane_pole}, Sekwencja gry: {sekwencja_gry})",
        fontsize=14, fontweight='bold')
    plt.ylabel("Siła sygnału (Projekcja na wektor 'Zajęte')", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    for i, txt in enumerate(wartosci):
        plt.annotate(f"{txt:.2f}", (etapy[i], wartosci[i]), textcoords="offset points", xytext=(0, 10), ha='center',
                     fontsize=12, fontweight='bold')

    print("\n--- WYNIKI DOWODU NA STRUMIEŃ RESIDUALNY ---")
    for etap, val in historia_strumienia:
        print(f"{etap}: {val:.3f}")

    plt.tight_layout()
    import os
    os.makedirs('../plots', exist_ok=True)
    sciezka_wykresu = '../plots/residual_stream_proof.png'
    plt.savefig(sciezka_wykresu, dpi=300)
    print(f"\nZapisano wykres do: {sciezka_wykresu}")
    plt.show()


if __name__ == "__main__":
    SCIEZKA_MODELU = '../models/transformer/tictactoe_model.pth'

    # Gramy partię, w której Pole 0 zostaje zajęte natychmiast na początku
    gra_testowa = [0, 4, 1, 8, 3]

    # Cel: Udowodnić, że sieć wie o tym, że Pole 0 jest zablokowane
    badaj_strumien_residualny(SCIEZKA_MODELU, gra_testowa, badane_pole=0)