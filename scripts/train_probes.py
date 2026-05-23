import torch
import torch.nn as nn

# --- IMPORTY Z NASZEGO NARZĘDZIOWNIKA ---
from probe_utils import MLPProbe, przygotuj_dane

# 1. Błyskawiczne ładowanie i podział danych (Ścieżka wychodzi folder wyżej: '../data/...')
print("Ładowanie danych za pomocą narzędziownika...")
aktywacje, relatywne_trening, relatywne_egzamin, _ = przygotuj_dane(
    sciezka_do_danych='../data/processed/dataset_pelny.pt',
    liczba_trening=1000,
    liczba_test=200
)

liczba_gier_trening = 1000
liczba_gier_egzamin = 200

for nazwa_warstwy in ['warstwa_0', 'warstwa_1', 'warstwa_2']:
    print(f"\n--- Rozpoczynanie treningu dla warstwy {nazwa_warstwy} ---")

    # Używamy poprawnej Sondy MLP z naszego modułu
    sonda = MLPProbe(128, 27)
    judge = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(sonda.parameters(), lr=0.005)

    aktywacje_warstwy = aktywacje[nazwa_warstwy]

    # Wyciągamy myśli z gotowego słownika i dzielimy na pule
    mysli_trening = aktywacje_warstwy[:liczba_gier_trening].view(-1, 128)
    mysli_egzamin = aktywacje_warstwy[liczba_gier_trening: liczba_gier_trening + liczba_gier_egzamin].view(-1, 128)

    epochs = 1000

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Uczymy się TYLKO na danych treningowych
        prediction = sonda(mysli_trening).view(-1, 3, 9)

        loss = judge(prediction, relatywne_trening)
        loss.backward()
        optimizer.step()

        # Sprawdzanie wyników w tle (Szukamy zjawiska Overfittingu)
        with torch.no_grad():
            chosen_states_trening = torch.argmax(prediction, dim=1)
            poprawne_trening = (chosen_states_trening == relatywne_trening).sum().item()
            skutecznosc_trening = (poprawne_trening / (liczba_gier_trening * 9 * 9)) * 100

            pred_egzamin = sonda(mysli_egzamin).view(-1, 3, 9)
            chosen_states_egzamin = torch.argmax(pred_egzamin, dim=1)
            poprawne_egzamin = (chosen_states_egzamin == relatywne_egzamin).sum().item()
            skutecznosc_egzamin = (poprawne_egzamin / (liczba_gier_egzamin * 9 * 9)) * 100

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoka: {epoch + 1:4d}/{epochs} | Trening: {skutecznosc_trening:.1f}% | EGZAMIN: {skutecznosc_egzamin:.1f}%")

# --- MODUŁ RENTGENA TERMINALOWEGO ---
print("\n" + "=" * 50)
print("URUCHAMIAM RENTGEN DLA OSTATNIEJ TRENOWANEJ WARSTWY (GRA Z EGZAMINU)")
print("=" * 50)

def pokaz_plansze_w_terminalu(sonda_model, mysli_test, relatywne_test, indeks_gry=0):
    sonda_model.eval()
    start = indeks_gry * 9
    koniec = start + 9

    mysli_jednej_gry = mysli_test[start:koniec]
    prawdziwa_plansza = relatywne_test[start:koniec]

    with torch.no_grad():
        przewidywania = sonda_model(mysli_jednej_gry).view(9, 3, 9)
        wyroki_sondy = torch.argmax(przewidywania, dim=1)

    symbole = {0: "⬜", 1: "🟦", 2: "🟥"}

    for krok in range(9):
        print(f"\n--- KROK {krok + 1} ---")
        print("RZECZYWISTOŚĆ        UMYSŁ SIECI (SONDA)")
        prawdziwy_krok = prawdziwa_plansza[krok].tolist()
        sonda_krok = wyroki_sondy[krok].tolist()

        for wiersz in range(3):
            p_w = " ".join([symbole[prawdziwy_krok[wiersz * 3 + i]] for i in range(3)])
            s_w = " ".join([symbole[sonda_krok[wiersz * 3 + i]] for i in range(3)])
            print(f"{p_w}   |   {s_w}")

pokaz_plansze_w_terminalu(sonda, mysli_egzamin, relatywne_egzamin, indeks_gry=0)