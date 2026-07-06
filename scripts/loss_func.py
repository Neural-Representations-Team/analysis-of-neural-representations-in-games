import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from probe_utils import LinearProbe, MLPProbe, przygotuj_dane, wizualizuj_geometrie_pca_3d


# --- NOWA FUNKCJA DO RYSOWANIA KRZYWYCH STRATY ---
def rysuj_krzywe_uczenia(train_losses, test_losses, train_accs, test_accs, epoki_x, nazwa_warstwy, typ_sondy):
    """
    Generuje i zapisuje wykresy Loss oraz Accuracy, aby udowodnić brak (lub obecność) overfittingu.
    """
    os.makedirs('../plots/learning_curves', exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Wykres Loss (Krzywa Straty)
    ax1.plot(epoki_x, train_losses, label='Strata Treningowa (Train Loss)', color='blue', alpha=0.7)
    ax1.plot(epoki_x, test_losses, label='Strata Egzaminacyjna (Test Loss)', color='red', alpha=0.7)
    ax1.set_title(f'Krzywa Straty - {typ_sondy} ({nazwa_warstwy})')
    ax1.set_xlabel('Epoka')
    ax1.set_ylabel('Loss (CrossEntropy)')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Wykres Accuracy (Skuteczność)
    ax2.plot(epoki_x, train_accs, label='Skuteczność Treningowa', color='blue', alpha=0.7)
    ax2.plot(epoki_x, test_accs, label='Skuteczność Egzaminacyjna', color='red', alpha=0.7)
    ax2.set_title(f'Krzywa Skuteczności - {typ_sondy} ({nazwa_warstwy})')
    ax2.set_xlabel('Epoka')
    ax2.set_ylabel('Skuteczność (%)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    sciezka = f'../plots/learning_curves/learning_curve_{nazwa_warstwy}_{typ_sondy}.png'
    plt.savefig(sciezka)
    plt.close()
    print(f"Zapisano wykresy uczenia: {sciezka}")


# --- GŁÓWNA CZĘŚĆ SKRYPTU ---

print("Ładowanie danych za pomocą narzędziownika...")
aktywacje, relatywne_trening, relatywne_egzamin, _ = przygotuj_dane(
    sciezka_do_danych='../data/processed/dataset_pelny.pt',
    liczba_trening=1000,
    liczba_test=200
)

liczba_gier_trening = 1000
liczba_gier_egzamin = 200

for nazwa_warstwy in ['warstwa_0', 'warstwa_1', 'warstwa_2']:
    print(f"\n{'=' * 50}")
    print(f"ROZPOCZYNAMY ANALIZĘ DLA: {nazwa_warstwy.upper()}")
    print(f"{'=' * 50}")

    aktywacje_warstwy = aktywacje[nazwa_warstwy]

    # Wyciągamy myśli z gotowego słownika i dzielimy na pule
    mysli_trening = aktywacje_warstwy[:liczba_gier_trening].view(-1, 128)
    mysli_egzamin = aktywacje_warstwy[liczba_gier_trening: liczba_gier_trening + liczba_gier_egzamin].view(-1, 128)

    # --- PĘTLA PO TYPACH SOND (LINIOWA I NIELINIOWA) ---
    for typ_sondy, klasa_sondy in [('Liniowa', LinearProbe), ('Nieliniowa', MLPProbe)]:
        print(f"\n--- Trening sondy: {typ_sondy} ---")

        # Inicjalizacja odpowiedniej sondy
        sonda = klasa_sondy(128, 27)
        judge = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(sonda.parameters(), lr=0.005)

        epochs = 1000

        # Listy do przechowywania historii uczenia
        historia_train_loss = []
        historia_test_loss = []
        historia_train_acc = []
        historia_test_acc = []
        epoki_zapisane = []

        for epoch in range(epochs):
            sonda.train()
            optimizer.zero_grad()

            # Uczymy się TYLKO na danych treningowych
            prediction_train = sonda(mysli_trening).view(-1, 3, 9)
            loss_train = judge(prediction_train, relatywne_trening)

            loss_train.backward()
            optimizer.step()

            # Zapisujemy metryki co 10 epok, by mieć płynny wykres bez zamulania pamięci
            if epoch % 10 == 0 or epoch == epochs - 1:
                sonda.eval()
                with torch.no_grad():
                    # Strata i skuteczność TRENINGOWA
                    chosen_states_trening = torch.argmax(prediction_train, dim=1)
                    poprawne_trening = (chosen_states_trening == relatywne_trening).sum().item()
                    skutecznosc_trening = (poprawne_trening / (liczba_gier_trening * 9 * 9)) * 100

                    # Strata i skuteczność EGZAMINACYJNA (Testowa)
                    prediction_test = sonda(mysli_egzamin).view(-1, 3, 9)
                    loss_test = judge(prediction_test, relatywne_egzamin)

                    chosen_states_egzamin = torch.argmax(prediction_test, dim=1)
                    poprawne_egzamin = (chosen_states_egzamin == relatywne_egzamin).sum().item()
                    skutecznosc_egzamin = (poprawne_egzamin / (liczba_gier_egzamin * 9 * 9)) * 100

                    # Logowanie do historii
                    historia_train_loss.append(loss_train.item())
                    historia_test_loss.append(loss_test.item())
                    historia_train_acc.append(skutecznosc_trening)
                    historia_test_acc.append(skutecznosc_egzamin)
                    epoki_zapisane.append(epoch)

            # Wypisywanie w konsoli rzadziej (co 100 epok) żeby nie śmiecić
            if epoch % 100 == 0 or epoch == epochs - 1:
                print(
                    f"Epoka: {epoch + 1:4d}/{epochs} | Trening Acc: {skutecznosc_trening:.1f}% (Loss: {loss_train.item():.3f}) | EGZAMIN Acc: {skutecznosc_egzamin:.1f}% (Loss: {loss_test.item():.3f})")

        # --- GENEROWANIE WYKRESÓW PO TRENINGU SONDY ---
        rysuj_krzywe_uczenia(
            historia_train_loss, historia_test_loss,
            historia_train_acc, historia_test_acc,
            epoki_zapisane, nazwa_warstwy, typ_sondy
        )

        # --- WYCIĄGANIE WAG DO WIZUALIZACJI ---
        if typ_sondy == 'Liniowa':
            wagi_koncowe = sonda.layer.weight.data
        else:
            wagi_koncowe = sonda.layer[2].weight.data

        # Odpalenie generowania wykresu interaktywnego 3D
        wizualizuj_geometrie_pca_3d(wagi_koncowe, tytul=f"Geometria Sieci - {typ_sondy} {nazwa_warstwy}")

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