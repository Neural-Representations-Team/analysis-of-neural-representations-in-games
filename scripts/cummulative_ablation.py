import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
import os
import numpy as np

# --- IMPORTY Z TWOJEGO NARZĘDZIOWNIKA ---
# Zakładam, że te klasy masz w probe_utils.py. Jeśli nie, po prostu upewnij się,
# że struktura TinyTicTacToeGPT i LinearProbe zgadza się z Twoją bazą kodu.
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


# --- GŁÓWNA LOGIKA ---
def uruchom_eksperyment_skumulowany():
    print("1. Ładowanie danych i czystych aktywacji...")
    # Ścieżki - dostosuj jeśli masz inną strukturę folderów!
    SCIEZKA_MODELU = '../models/transformer/tictactoe_model.pth'
    SCIEZKA_DANYCH_JSON = '../data/games.json'

    aktywacje, relatywne_trening, relatywne_test, _ = przygotuj_dane(
        sciezka_do_danych='../data/processed/dataset_pelny.pt',
        liczba_trening=1000,
        liczba_test=200
    )

    mysli_trening_L1 = aktywacje['warstwa_1'][:1000].view(-1, 128)

    print("2. Szybki trening bazowej Sondy Liniowej dla Fizyki Planszy (Warstwa L1)...")
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

    print("\n3. Pobieranie sekwencji testowych do przepuszczania przez uszkodzony model...")
    with open(SCIEZKA_DANYCH_JSON, 'r') as f:
        wszystkie_gry = json.load(f)

    gry_testowe = wszystkie_gry[1000:1200]
    tensory_gier = torch.tensor([get_padded_sequence(gra) for gra in gry_testowe], dtype=torch.long)

    # Funkcja do przepuszczenia gier przez uszkodzony model i wyciągnięcia aktywacji L1
    def pobierz_zepsute_aktywacje(uszkodzony_model):
        zepsute_L1 = []

        def hook(mod, inp, out):
            zepsute_L1.append(out.detach())

        handle = uszkodzony_model.transformer.layers[1].register_forward_hook(hook)
        with torch.no_grad():
            uszkodzony_model(tensory_gier)
        handle.remove()
        return zepsute_L1[0].view(-1, 128)

    print("\n4. Pojedyncza ablacja głów w L1 w celu ustalenia hierarchii ważności...")
    ranking_glow = []

    for h in range(8):
        # Za każdym razem ładujemy świeży model, by zepsuć tylko 1 głowę
        model = TinyTicTacToeGPT(d_model=128, num_layers=3, nhead=8)
        model.load_state_dict(torch.load(SCIEZKA_MODELU, map_location='cpu', weights_only=True))
        model.eval()

        wyzeruj_glowy(model, warstwa_idx=1, lista_glow=[h])
        zepsute_akty = pobierz_zepsute_aktywacje(model)
        acc = evaluate_physics_accuracy(sonda, zepsute_akty, relatywne_test)
        spadek = baseline_acc - acc
        ranking_glow.append({'glowa': h, 'acc': acc, 'spadek': spadek})
        print(f" - Ablacja głowy L1.G{h}: spadek o {spadek:.2f}% (Acc: {acc:.1f}%)")

    # Sortujemy od największego spadku do najmniejszego
    ranking_glow.sort(key=lambda x: x['spadek'], reverse=True)
    posortowane_glowy = [item['glowa'] for item in ranking_glow]

    print(f"\nHierarchia ważności głów (od najważniejszej): {posortowane_glowy}")

    print("\n5. ABLACJA SKUMULOWANA (Cumulative Knockout)...")
    wyniki_skumulowane = [baseline_acc]  # Zaczynamy od 0 wyłączonych głów
    odciete_glowy = []

    for h in posortowane_glowy:
        odciete_glowy.append(h)
        model = TinyTicTacToeGPT(d_model=128, num_layers=3, nhead=8)
        model.load_state_dict(torch.load(SCIEZKA_MODELU, map_location='cpu', weights_only=True))
        model.eval()

        wyzeruj_glowy(model, warstwa_idx=1, lista_glow=odciete_glowy)
        zepsute_akty = pobierz_zepsute_aktywacje(model)
        acc = evaluate_physics_accuracy(sonda, zepsute_akty, relatywne_test)
        wyniki_skumulowane.append(acc)

        print(f" Wyłączono {len(odciete_glowy)} głów {odciete_glowy} -> Skuteczność fizyki: {acc:.1f}%")

    print("\n6. Generowanie wykresu naukowego...")
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(9, 6))

    os_x = np.arange(9)
    ax.plot(os_x, wyniki_skumulowane, marker='o', color='#e74c3c', linewidth=3, markersize=8, zorder=3)

    # Obszar losowego zgadywania (~33%)
    ax.axhline(33.3, color='gray', linestyle='--', linewidth=2, label='Poziom losowego zgadywania (~33%)', zorder=1)

    ax.set_title('Ablacja Skumulowana w Warstwie L1\n(Zjawisko Graceful Degradation)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Liczba wyłączonych Głów Uwagi (od najważniejszej do najmniej ważnej)', fontsize=12,
                  fontweight='bold')
    ax.set_ylabel('Skuteczność detekcji fizyki planszy (%)', fontsize=12, fontweight='bold')

    ax.set_xticks(os_x)
    ax.set_ylim(20, 105)
    ax.legend(loc='lower left')

    # Dodanie adnotacji wartości nad punktami
    for i, txt in enumerate(wyniki_skumulowane):
        ax.annotate(f"{txt:.1f}%", (os_x[i], wyniki_skumulowane[i] + 2), ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    os.makedirs('../plots', exist_ok=True)
    plt.savefig('../plots/08_ablacja_skumulowana.png', dpi=300)
    print("Zapisano wykres do '../plots/08_ablacja_skumulowana.png'!")
    plt.show()


if __name__ == "__main__":
    uruchom_eksperyment_skumulowany()