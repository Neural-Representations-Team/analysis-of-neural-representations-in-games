import torch.nn as nn
import torch

class LinearProbe(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Zamiast jednej prostej rury, dodajemy stację pośrednią (Hidden Layer)
        hidden_size = 256  # Rozszerzamy umysł sondy

        self.layer = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.ReLU(),  # Funkcja aktywacji - daje sondzie zdolność wyginania linii
            nn.Linear(hidden_size, self.output_size)
        )

        # self.layer = nn.Linear(self.input_size, self.output_size)

    def forward(self, x):
        return self.layer(x)

dane_z_dysku = torch.load('data/processed/dataset_pelny.pt')
slownik_aktywacji = dane_z_dysku['aktywacje']  # Nasz materiał dowodowy
plansze = dane_z_dysku['plansze']      # Nasz klucz odpowiedzi

liczba_gier = plansze.size(0)
liczba_krokow = plansze.size(1)


for nazwa_warstwy in ['warstwa_0', 'warstwa_1', 'warstwa_2']:
    print(f"\n--- Rozpoczynanie treningu dla warstwy {nazwa_warstwy} ---")

    sonda = LinearProbe(128, 27)
    judge = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(sonda.parameters(), lr=0.005)

    aktywacje = slownik_aktywacji[nazwa_warstwy]

    wszystkie_mysli = aktywacje.view(-1, 128)
    wszystkie_stany_planszy = plansze.view(-1, 9)

    epochs = 2000

    for epoch in range(epochs):
        optimizer.zero_grad()

        prediction = sonda(wszystkie_mysli)

        prediction = prediction.view(-1, 3, 9)

        real_plane = wszystkie_stany_planszy % 3

        liczba_jedynek = (real_plane == 1).sum(dim=1, keepdim=True)
        liczba_dwojek = (real_plane == 2).sum(dim=1, keepdim=True)

        ruch_dwojki = liczba_jedynek > liczba_dwojek

        relatywna_plansza = real_plane.clone()
        maska_ruchu = ruch_dwojki.expand_as(real_plane)

        relatywna_plansza[maska_ruchu & (real_plane == 1)] = 2
        relatywna_plansza[maska_ruchu & (real_plane == 2)] = 1

        loss = judge(prediction, relatywna_plansza)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            chosen_states = torch.argmax(prediction, dim=1)
            poprawne_trafienia = (chosen_states == relatywna_plansza).sum().item()
            wszystkie_pola = liczba_gier * liczba_krokow * 9
            skutecznosc = (poprawne_trafienia / wszystkie_pola) * 100

        if epoch % 200 == 0 or epoch == epochs - 1:
            print(f"Epoka: {epoch + 1:3d}/100 | Błąd: {loss.item():.4f} | Skuteczność: {skutecznosc:.1f}%")

# --- MODUŁ WIZUALIZACJI RENTGENOWSKIEJ ---
print("\n" + "=" * 50)
print("URUCHAMIAM RENTGEN DLA OSTATNIEJ TRENOWANEJ WARSTWY")
print("=" * 50)


def pokaz_plansze_w_terminalu(sonda, wszystkie_mysli, wszystkie_stany_plansz, indeks_gry=0):
    # Wyłączamy losowe procesy w sondzie na czas testu
    sonda.eval()

    # Wyciągamy dane tylko dla jednej, wybranej gry (9 kroków)
    start = indeks_gry * 9
    koniec = start + 9

    mysli_jednej_gry = wszystkie_mysli[start:koniec]
    plansze_jednej_gry = wszystkie_stany_plansz[start:koniec] % 3

    # Musimy nałożyć Logikę Relatywną na tę jedną grę, żeby mieć poprawny klucz odpowiedzi
    liczba_jedynek = (plansze_jednej_gry == 1).sum(dim=1, keepdim=True)
    liczba_dwojek = (plansze_jednej_gry == 2).sum(dim=1, keepdim=True)
    ruch_dwojki = liczba_jedynek > liczba_dwojek

    relatywna_plansza = plansze_jednej_gry.clone()
    maska = ruch_dwojki.expand_as(plansze_jednej_gry)
    relatywna_plansza[maska & (plansze_jednej_gry == 1)] = 2
    relatywna_plansza[maska & (plansze_jednej_gry == 2)] = 1

    # Przepuszczamy myśli przez sondę
    with torch.no_grad():
        przewidywania = sonda(mysli_jednej_gry)
        przewidywania = przewidywania.view(9, 3, 9)
        wyroki_sondy = torch.argmax(przewidywania, dim=1)

    # Legenda znaczków do terminala
    # ⬜ - Puste pole
    # 🟦 - Mój pionek (Ten, który teraz wykonuje ruch)
    # 🟥 - Pionek wroga
    symbole = {0: "⬜", 1: "🟦", 2: "🟥"}

    for krok in range(9):
        print(f"\n--- KROK {krok + 1} ---")
        print("RZECZYWISTOŚĆ        UMYSŁ SIECI (SONDA)")

        prawdziwy_krok = relatywna_plansza[krok].tolist()
        sonda_krok = wyroki_sondy[krok].tolist()

        for wiersz in range(3):
            # Rysowanie prawdziwej planszy
            p_w = " ".join([symbole[prawdziwy_krok[wiersz * 3 + i]] for i in range(3)])
            # Rysowanie tego, co zgadła sonda
            s_w = " ".join([symbole[sonda_krok[wiersz * 3 + i]] for i in range(3)])

            print(f"{p_w}   |   {s_w}")


# Uruchamiamy wizualizację dla pierwszej gry z brzegu (indeks 0)
pokaz_plansze_w_terminalu(sonda, wszystkie_mysli, wszystkie_stany_planszy, indeks_gry=0)