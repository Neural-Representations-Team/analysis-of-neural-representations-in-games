import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- IMPORTY Z NASZEGO NARZĘDZIOWNIKA ---
from probe_utils import przygotuj_dane, trenuj_sondy

# Stylizacja wykresów na profesjonalny raport naukowy
plt.style.use('ggplot')


# --- FUNKCJE RYSUJĄCE WIZUALIZACJĘ ---

def save_detailed_heatmap(predictions, prawdziwa_gra_egzamin_L, nazwa_warstwy, probed_object, game_length, example_id):
    # Zmieniamy płaskie [9, 27] z powrotem na blok [9, 3, 9]
    predictions = predictions.view(9, 3, 9)
    procenty = torch.softmax(predictions, dim=1)

    fig, axes = plt.subplots(1, 9, figsize=(20, 3.5), constrained_layout=True)
    tytul = f"{probed_object} - {nazwa_warstwy.upper()} - Długość: {game_length} - Przykład Egzamin: #{example_id}"
    fig.suptitle(tytul, fontsize=16, fontweight='bold')

    slownik_nazw = {0: "Puste", 1: "Moje", 2: "Wroga"}

    for krok in range(9):
        ax = axes[krok]

        # Oceniamy plansze tylko tam gdzie faktycznie trwała gra (krok < game_length)
        if krok < game_length:
            real_ground_truth_3x3 = prawdziwa_gra_egzamin_L[krok].numpy().reshape(3, 3)

            procenty_krok = procenty[krok]  # [3, 9]
            wybrane_klasy = torch.argmax(procenty_krok, dim=0)  # [9]

            pewnosc = procenty_krok[wybrane_klasy, torch.arange(9)] * 100

            pewnosc_3x3 = pewnosc.numpy().reshape(3, 3)
            klasy_3x3 = wybrane_klasy.numpy().reshape(3, 3)

            podpisy = np.empty((3, 3), dtype=object)
            for i in range(3):
                for j in range(3):
                    zgadnieta = slownik_nazw[klasy_3x3[i, j]]
                    procent_liczba = pewnosc_3x3[i, j]

                    if klasy_3x3[i, j] != real_ground_truth_3x3[i, j]:
                        podpisy[i, j] = f"BŁĄD!\n{zgadnieta}\n{procent_liczba:.0f}%"
                    else:
                        podpisy[i, j] = f"{zgadnieta}\n{procent_liczba:.0f}%"

            sns.heatmap(pewnosc_3x3, annot=podpisy, fmt="", cmap="Blues",
                        cbar=False, ax=ax, vmin=33, vmax=100, linewidths=1, linecolor='black')
        else:
            # Poza granicami gry, plansza jest pusta. Sonda klasyfikuje puste pola (Padding).
            pewnosc_3x3 = np.zeros((3, 3))
            podpisy = np.empty((3, 3), dtype=object)
            for i in range(3):
                for j in range(3):
                    podpisy[i, j] = "Puste\n(PADDING)"

            sns.heatmap(pewnosc_3x3, annot=podpisy, fmt="", cmap="Greys", cbar=False, ax=ax, linewidths=1,
                        linecolor='black')

        ax.set_title(f"Ruch {krok + 1}")
        ax.set_xticks([])
        ax.set_yticks([])

    return fig


def save_real_board_detailed(prawdziwa_gra_L, tytul, game_length):
    fig, axes = plt.subplots(1, 9, figsize=(20, 3.5), constrained_layout=True)
    fig.suptitle(tytul, fontsize=16, fontweight='bold')
    slownik_nazw = {0: "Puste", 1: "Moje", 2: "Wroga"}

    for krok in range(9):
        ax = axes[krok]

        if krok < game_length:
            prawda_3x3 = prawdziwa_gra_L[krok].numpy().reshape(3, 3)
            podpisy = np.empty((3, 3), dtype=object)
            for i in range(3):
                for j in range(3):
                    podpisy[i, j] = slownik_nazw[prawda_3x3[i, j]]

            sns.heatmap(prawda_3x3, annot=podpisy, fmt="", cmap="Greens",
                        cbar=False, ax=ax, vmin=0, vmax=2, linewidths=1, linecolor='black')
        else:
            pewnosc_3x3 = np.zeros((3, 3))
            podpisy = np.empty((3, 3), dtype=object)
            for i in range(3):
                for j in range(3):
                    podpisy[i, j] = "Puste\n(PADDING)"
            sns.heatmap(pewnosc_3x3, annot=podpisy, fmt="", cmap="Greys", cbar=False, ax=ax, linewidths=1,
                        linecolor='black')

        ax.set_title(f"Ruch {krok + 1}")
        ax.set_xticks([])
        ax.set_yticks([])
    return fig


def save_and_close_figure(fig, filename):
    # ZAKTUALIZOWANA ŚCIEŻKA: Zapisujemy do folderu ../plots/detailed_heatmaps/
    directory = '../plots/detailed_heatmaps/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    path = os.path.join(directory, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"Zapisano wykres diagnostyczny: {path}")


# --- GŁÓWNY SILNIK ANALITYCZNY ---

if __name__ == "__main__":
    liczba_trening = 1000
    liczba_egzamin = 200

    print("Ładowanie danych diagnostycznych za pomocą narzędziownika...")
    aktywacje, relatywne_trening, relatywne_egzamin, dlugosci_egzamin = przygotuj_dane(
        sciezka_do_danych='../data/processed/dataset_pelny.pt',
        liczba_trening=liczba_trening,
        liczba_test=liczba_egzamin
    )

    # --- SELEKCJA PRZYKŁADÓW EGZAMINACYJNYCH WG DŁUGOŚCI ---
    examples_by_length = {L: [] for L in range(5, 10)}
    for i, dlugosc in enumerate(dlugosci_egzamin):
        examples_by_length[dlugosc.item()].append(i)

    print("\nOSTRZEŻENIE: Ten skrypt wygeneruje i zapisze 35 PNG wykresów diagnostycznych.")
    print("Zajmie to około kilkunastu minut. Użyjemy rygorystycznego treningu 1000 epok.")
    print("Pliki zostaną zapisane w katalogu: ../plots/detailed_heatmaps/")
    print("\nRozpoczynam rygorystyczny trening 1000 epok diagnostyczny sond...")

    target_lengths = [5, 6, 7, 8, 9]

    # Słownik do przechowywania wytrenowanych sond
    trained_sondas = {warstwa: {'linear': None, 'mlp': None} for warstwa in ['warstwa_0', 'warstwa_1', 'warstwa_2']}

    # Szybki trening dla każdej warstwy
    for warstwa in ['warstwa_0', 'warstwa_1', 'warstwa_2']:
        print(f"  > Nauka diagnostyczna dla {warstwa.upper()} (1000 epok)...")

        sonda_liniowa, sonda_mlp = trenuj_sondy(
            aktywacje_warstwy=aktywacje[warstwa],
            relatywne_trening=relatywne_trening,
            liczba_trening=liczba_trening,
            epochs=1000
        )

        trained_sondas[warstwa]['linear'] = sonda_liniowa
        trained_sondas[warstwa]['mlp'] = sonda_mlp
        print(f"  Zakończono naukę diagnostyczną dla: {warstwa.upper()}.")

    print("\nGenerowanie ostatecznych wykresów diagnostycznych dla egzaminu...")

    # Przygotowanie macierzy 3D do wyciągania pojedynczych klatek
    relatywne_egzamin_3d = relatywne_egzamin.view(-1, 9, 9)

    for L in target_lengths:
        example_indices = examples_by_length[L]
        if not example_indices:
            print(f"\nPOMIJAM DETALICZNE HEATMAPY: Brak gier o długości {L} ruchów w egzaminacyjnej paczce.")
            continue

        example_id = example_indices[0]
        print(f"\n--- Przetwarzanie egzaminacyjnej gry #{example_id} (Długość: {L} ruchów) ---")

        prawdziwa_gra_egzamin_L = relatywne_egzamin_3d[example_id]

        fig_real = save_real_board_detailed(prawdziwa_gra_egzamin_L, f"STAN REALNY PLANSZY (EGZAMIN) - Długość: {L}", L)
        save_and_close_figure(fig_real, f"L{L}_real.png")

        for warstwa in ['warstwa_0', 'warstwa_1', 'warstwa_2']:
            mysli_gra_egzamin_L = aktywacje[warstwa][liczba_trening + example_id].view(9, 128)

            sonda_lin = trained_sondas[warstwa]['linear']
            sonda_lin.eval()
            with torch.no_grad():
                pred_lin_L = sonda_lin(mysli_gra_egzamin_L)
            fig_lin = save_detailed_heatmap(pred_lin_L, prawdziwa_gra_egzamin_L, warstwa, "Sonda LINIOWA (Fizyka)", L,
                                            example_id)
            save_and_close_figure(fig_lin, f"L{L}_{warstwa}_linear.png")

            sonda_mlp = trained_sondas[warstwa]['mlp']
            sonda_mlp.eval()
            with torch.no_grad():
                pred_mlp_L = sonda_mlp(mysli_gra_egzamin_L)
            fig_mlp = save_detailed_heatmap(pred_mlp_L, prawdziwa_gra_egzamin_L, warstwa, "Sonda MLP (Taktyka)", L,
                                            example_id)
            save_and_close_figure(fig_mlp, f"L{L}_{warstwa}_mlp.png")

    print(
        "\nGotowe. Wszystkie 35 ostatecznych wykresów diagnostycznych zostało zapisanych w: '../plots/detailed_heatmaps/'.")