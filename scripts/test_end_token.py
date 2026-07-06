import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

def test_wizualny_konca_gry(model, sekwencja_ruchow, device='cpu'):
    """
    Wizualizuje stan planszy oraz przewidywania modelu dla kolejnego ruchu,
    ze szczególnym uwzględnieniem tokenu końca gry (9).
    """
    model.eval()

    # 1. Odtworzenie stanu planszy do wizualizacji
    plansza = np.zeros(9)  # 0 = puste, 1 = X, 2 = O
    gracz = 1
    for ruch in sekwencja_ruchow:
        if ruch < 9:
            plansza[ruch] = gracz
            gracz = 2 if gracz == 1 else 1

    # 2. Przygotowanie wejścia dla modelu (z paddingiem)
    padded_seq = sekwencja_ruchow + [10] * (10 - len(sekwencja_ruchow))
    inputs = torch.tensor(padded_seq).unsqueeze(0).to(device)

    # 3. Przepuszczenie przez model
    with torch.no_grad():
        logits = model(inputs)
        step_logits = logits[0, len(sekwencja_ruchow) - 1]
        probs = torch.softmax(step_logits, dim=0).cpu().numpy()

    # --- WIZUALIZACJA ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Analiza Decyzji Modelu\nSekwencja: {sekwencja_ruchow}', fontsize=16, fontweight='bold')

    # Wykres 1: Plansza
    ax1.set_title('Aktualny stan planszy', fontsize=12)
    ax1.set_xlim(-0.5, 2.5)
    ax1.set_ylim(-0.5, 2.5)
    ax1.invert_yaxis()  # Żeby pole 0 było na górze
    ax1.axis('off')

    # Rysowanie siatki
    for i in range(1, 3):
        ax1.axhline(i - 0.5, color='black', linewidth=2)
        ax1.axvline(i - 0.5, color='black', linewidth=2)

    # Rysowanie znaków
    symbole = {0: '', 1: 'X', 2: 'O'}
    kolory = {1: '#2980b9', 2: '#c0392b'}
    for i in range(9):
        row, col = i // 3, i % 3
        znak = symbole[plansza[i]]
        if znak:
            ax1.text(col, row, znak, fontsize=40, ha='center', va='center',
                     color=kolory[plansza[i]], fontweight='bold')
        else:
            ax1.text(col, row, str(i), fontsize=12, ha='center', va='center', color='gray', alpha=0.5)

    # Wykres 2: Prawdopodobieństwa (Logity)
    ax2.set_title('Prawdopodobieństwo kolejnego tokenu', fontsize=12)
    klasy = [f'Pole {i}' for i in range(9)] + ['Koniec (9)', 'Pad (10)']

    # Ustalanie kolorów słupków
    kolory_slupkow = []
    for i in range(11):
        if i == 9:
            kolory_slupkow.append('#27ae60')  # Zielony dla końca gry
        elif i < 9 and plansza[i] != 0:
            kolory_slupkow.append('#7f8c8d')  # Szary dla zajętych
        elif i == 10:
            kolory_slupkow.append('#bdc3c7')
        else:
            kolory_slupkow.append('#3498db')  # Niebieski dla legalnych

    y_pos = np.arange(len(klasy))
    bars = ax2.barh(y_pos, probs * 100, color=kolory_slupkow)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(klasy)
    ax2.set_xlabel('Prawdopodobieństwo (%)')
    ax2.set_xlim(0, 100)
    ax2.invert_yaxis()

    # Dodanie procentów na słupkach
    for bar in bars:
        width = bar.get_width()
        if width > 0.5:
            ax2.text(width + 1, bar.get_y() + bar.get_height() / 2,
                     f'{width:.1f}%', va='center', fontsize=10)

    plt.tight_layout()
    plt.show()


# --- PRZYKŁADY DO PRZETESTOWANIA ---


# 1. Musisz mieć tę samą klasę w pliku lub zaimportowaną
# Upewnij się, że klasa TinyTicTacToeGPT jest zdefiniowana w tym pliku
# lub zaimportowana z pliku, gdzie ją trzymasz.

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

# 2. Inicjalizacja modelu (parametry muszą być identyczne jak przy treningu!)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TinyTicTacToeGPT(d_model=128, num_layers=3, nhead=8).to(device)

# 3. Wczytanie wytrenowanych wag (podaj poprawną ścieżkę!)
model_path = '../models/transformer/tictactoe_model.pth'
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

# Dopiero teraz możesz wywołać test:
test_wizualny_konca_gry(model, [4, 0, 8, 1], device=device)

# 1. Gra w toku (brak wygranej, są wolne pola)
# Oczekiwane: równe szanse na puste pola, Koniec(9) blisko 0%
test_wizualny_konca_gry(model, [4, 0, 8, 1])

# 2. Wygrana w 7. ruchu (Obwód Sędziego ucina grę)
# Oczekiwane: Koniec(9) = 99.9%, pomimo że są wolne pola
test_wizualny_konca_gry(model, [0, 1, 4, 8, 3, 5, 6])

# 3. Pełna plansza - Remis (Obwód Mapy wymusza koniec)
# Oczekiwane: Koniec(9) = 99.9%, bo nie ma pustych pól
test_wizualny_konca_gry(model, [1, 0, 3, 2, 4, 5, 6, 7, 8])