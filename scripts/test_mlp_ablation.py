import torch
import torch.nn as nn
import json

# --- IMPORTY Z TWOJEGO NARZĘDZIOWNIKA ---
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
        skutecznosc_fizyki = ((wyroki == 0) == (prawdziwe_plansze == 0)).float().mean().item() * 100
    return skutecznosc_fizyki


def wyzeruj_glowy(model, warstwa_idx):
    """
    Zeruje wagi wyjściowe ze wszystkich głów uwagi (Attention) w danej warstwie.
    """
    with torch.no_grad():
        model.transformer.layers[warstwa_idx].self_attn.out_proj.weight[:] = 0.0


def wyzeruj_mlp(model, warstwa_idx):
    """
    Zeruje wagi warstwy gęstej (Feed-Forward / MLP) wewnątrz danego bloku Transformera.
    W architekturze PyTorch odpowiada to liniowej warstwie `linear2`.
    """
    with torch.no_grad():
        model.transformer.layers[warstwa_idx].linear2.weight[:] = 0.0


# --- LOGIKA ABLACJI MLP ---
def uruchom_ablacje_mlp():
    print("1. Ładowanie danych...")
    SCIEZKA_MODELU = '../models/transformer/tictactoe_model.pth'
    SCIEZKA_DANYCH_JSON = '../data/games.json'

    aktywacje, relatywne_trening, relatywne_test, _ = przygotuj_dane(
        sciezka_do_danych='../data/processed/dataset_pelny.pt',
        liczba_trening=1000,
        liczba_test=200
    )

    mysli_trening_L1 = aktywacje['warstwa_1'][:1000].view(-1, 128)

    print("2. Trening referencyjnej Sondy Liniowej na nietkniętej warstwie L1...")
    sonda = LinearProbe(128, 27)
    optimizer = torch.optim.Adam(sonda.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1000):
        optimizer.zero_grad()
        loss = criterion(sonda(mysli_trening_L1).view(-1, 3, 9), relatywne_trening)
        loss.backward()
        optimizer.step()

    baseline_acc = evaluate_physics_accuracy(sonda, aktywacje['warstwa_1'][1000:1200].view(-1, 128), relatywne_test)
    print(f"--> [BAZA] Czysta skuteczność fizyki (Baseline L1): {baseline_acc:.1f}%\n")

    print("3. Przygotowywanie danych do pomiaru awarii...")
    with open(SCIEZKA_DANYCH_JSON, 'r') as f:
        wszystkie_gry = json.load(f)

    gry_testowe = wszystkie_gry[1000:1200]
    tensory_gier = torch.tensor([get_padded_sequence(gra) for gra in gry_testowe], dtype=torch.long)

    def pobierz_aktywacje_l1(uszkodzony_model):
        zepsute_L1 = []

        def hook(mod, inp, out):
            zepsute_L1.append(out.detach())

        handle = uszkodzony_model.transformer.layers[1].register_forward_hook(hook)
        with torch.no_grad():
            uszkodzony_model(tensory_gier)
        handle.remove()
        return zepsute_L1[0].view(-1, 128)

    print("4. EKSPERYMENT: Ablacja Komponentów w L1...")

    # Scenariusz A: Wyłączamy całą sieć uwagi (Attention) w L1
    model_attn_off = TinyTicTacToeGPT(d_model=128, num_layers=3, nhead=8)
    model_attn_off.load_state_dict(torch.load(SCIEZKA_MODELU, map_location='cpu', weights_only=True))
    model_attn_off.eval()

    wyzeruj_glowy(model_attn_off, warstwa_idx=1)
    acc_attn_off = evaluate_physics_accuracy(sonda, pobierz_aktywacje_l1(model_attn_off), relatywne_test)
    print(f" - [TEST A] Skuteczność bez Głów Uwagi w L1: {acc_attn_off:.1f}%")

    # Scenariusz B: Wyłączamy samą warstwę MLP (Feed-Forward) w L1
    model_mlp_off = TinyTicTacToeGPT(d_model=128, num_layers=3, nhead=8)
    model_mlp_off.load_state_dict(torch.load(SCIEZKA_MODELU, map_location='cpu', weights_only=True))
    model_mlp_off.eval()

    wyzeruj_mlp(model_mlp_off, warstwa_idx=1)
    acc_mlp_off = evaluate_physics_accuracy(sonda, pobierz_aktywacje_l1(model_mlp_off), relatywne_test)
    print(f" - [TEST B] Skuteczność bez warstwy MLP w L1: {acc_mlp_off:.1f}%")

    # Scenariusz C: Zrównujemy z ziemią cały blok L1 (Attention + MLP)
    model_all_off = TinyTicTacToeGPT(d_model=128, num_layers=3, nhead=8)
    model_all_off.load_state_dict(torch.load(SCIEZKA_MODELU, map_location='cpu', weights_only=True))
    model_all_off.eval()

    wyzeruj_glowy(model_all_off, warstwa_idx=1)
    wyzeruj_mlp(model_all_off, warstwa_idx=1)
    acc_all_off = evaluate_physics_accuracy(sonda, pobierz_aktywacje_l1(model_all_off), relatywne_test)
    print(f" - [TEST C] Skuteczność po zniszczeniu całego bloku L1 (Attn + MLP): {acc_all_off:.1f}%\n")

    print("5. WNIOSKI:")
    if acc_mlp_off < acc_attn_off:
        print(" -> Warstwa MLP w L1 pełni krytyczną rolę w przechowywaniu mapy (pamięć modelu).")
    else:
        print(" -> Głowy Uwagi są ważniejsze niż MLP do poprawnego renderowania stanu planszy.")


if __name__ == "__main__":
    uruchom_ablacje_mlp()