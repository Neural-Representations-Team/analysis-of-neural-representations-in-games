import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# --- IMPORTY Z NASZEGO NARZĘDZIOWNIKA ---
from probe_utils import przygotuj_dane, trenuj_sondy

plt.style.use('ggplot')


def zapisz_wykres(fig, nazwa_pliku):
    if not os.path.exists('../plots'):
        os.makedirs('../plots')
    sciezka = os.path.join('../plots', nazwa_pliku)
    # Zapisujemy z parametrem tight, żeby nie ucięło legendy na dole
    fig.savefig(sciezka, dpi=300, bbox_inches='tight')
    print(f"Zapisano: {sciezka}")
    plt.close(fig)


def plot_probe_comparison(linear_acc, mlp_acc):
    warstwy = ['Warstwa 0', 'Warstwa 1', 'Warstwa 2']
    x = np.arange(len(warstwy))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.bar(x - width / 2, linear_acc, width, label='Sonda Liniowa (Płaska)', color='#3498db')
    ax.bar(x + width / 2, mlp_acc, width, label='Sonda MLP (Nieliniowa)', color='#e74c3c')
    ax.set_ylabel('Skuteczność na EGZAMINIE (%)', fontweight='bold')
    fig.suptitle('Zatarcie Liniowe: Detekcja kolorów pionków', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(warstwy)
    ax.set_ylim(50, 100)
    ax.legend()
    for i, v in enumerate(linear_acc): ax.text(i - width / 2, v + 0.5, f"{v}%", ha='center')
    for i, v in enumerate(mlp_acc): ax.text(i + width / 2, v + 0.5, f"{v}%", ha='center')
    return fig


def plot_world_model_resolution(fizyka_dane, taktyka_dane):
    warstwy = ['Warstwa 0', 'Warstwa 1', 'Warstwa 2']
    x = np.arange(len(warstwy))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.bar(x - width / 2, fizyka_dane, width, label='Fizyka Planszy (Puste vs Zajęte)', color='#2ecc71')
    ax.bar(x + width / 2, taktyka_dane, width, label='Własność Pionka (Mój vs Twój)', color='#9b59b6')
    ax.set_ylabel('Skuteczność na EGZAMINIE (%)', fontweight='bold')
    fig.suptitle('Dokładność Obrazu Świata (Sonda MLP)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(warstwy)
    ax.set_ylim(50, 105)
    ax.legend(loc='lower right')
    for i, v in enumerate(fizyka_dane): ax.text(i - width / 2, v + 0.5, f"{v}%", ha='center')
    for i, v in enumerate(taktyka_dane): ax.text(i + width / 2, v + 0.5, f"{v}%", ha='center')
    return fig


def plot_attention_shift_L9_grid(lin_fizyka, mlp_fizyka, lin_taktyka, mlp_taktyka):
    kolory = {
        'warstwa_0': '#e74c3c',
        'warstwa_1': '#2c3e50',
        'warstwa_2': '#3498db'
    }
    nazwy = {
        'warstwa_0': 'W0 (Początkowa)',
        'warstwa_1': 'W1 (Kartograf)',
        'warstwa_2': 'W2 (Decyzyjna)'
    }
    L = 9
    kroki = np.arange(1, L + 1)

    # Tworzymy matrycę 2x2 wykresów
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    fig.suptitle('Zatarcie reprezentacji w czasie dla pełnych partii (9 ruchów)', fontsize=16, fontweight='bold',
                 y=0.95)

    # Definiujemy co trafia na który wykres
    zestawienia = [
        (axes[0, 0], lin_fizyka, 'Fizyka (Puste vs Zajęte) - SONDA LINIOWA'),
        (axes[0, 1], mlp_fizyka, 'Fizyka (Puste vs Zajęte) - SONDA MLP'),
        (axes[1, 0], lin_taktyka, 'Taktyka (Mój vs Twój) - SONDA LINIOWA'),
        (axes[1, 1], mlp_taktyka, 'Taktyka (Mój vs Twój) - SONDA MLP')
    ]

    for ax, dane_dykt, tytul in zestawienia:
        for warstwa in ['warstwa_0', 'warstwa_1', 'warstwa_2']:
            if not dane_dykt[L][warstwa]:
                continue
            ax.plot(kroki, dane_dykt[L][warstwa], marker='o', markersize=6, linewidth=2.5,
                    label=nazwy[warstwa], color=kolory[warstwa])

        ax.set_title(tytul, fontsize=12)
        ax.set_ylim(30, 105)
        ax.set_xticks(kroki)
        ax.grid(True, linestyle='--', alpha=0.7)
        # Usuwamy górne i prawe krawędzie ramek dla nowoczesnego wyglądu
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Ustawiamy opisy osi tylko na skrajnych wykresach, aby uniknąć bałaganu
    for ax in axes[:, 0]:
        ax.set_ylabel('Skuteczność dekodowania (%)', fontweight='bold', fontsize=11)
    for ax in axes[1, :]:
        ax.set_xlabel('Numer ruchu w partii', fontweight='bold', fontsize=11)

    # Wspólna legenda na samym dole
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.02), fontsize=12)

    # Optymalizacja odstępów
    plt.subplots_adjust(bottom=0.12, hspace=0.15, wspace=0.1)

    return fig


# --- GŁÓWNY SILNIK ANALITYCZNY ---

if __name__ == "__main__":
    liczba_trening = 1000
    liczba_test = 200

    print("Ładowanie danych przy użyciu narzędziownika...")
    aktywacje, relatywne_trening, relatywne_test, dlugosci_test = przygotuj_dane(
        sciezka_do_danych='../data/processed/dataset_pelny.pt',
        liczba_trening=liczba_trening,
        liczba_test=liczba_test
    )

    lin_wyniki = []
    mlp_wyniki = []
    fizyka_dane_mlp = []
    taktyka_dane_mlp = []

    # Rozbudowane słowniki na statystyki
    struktura = lambda: {L: {w: [] for w in ['warstwa_0', 'warstwa_1', 'warstwa_2']} for L in range(5, 10)}
    dane_lin_fizyka = struktura()
    dane_lin_taktyka = struktura()
    dane_mlp_fizyka = struktura()
    dane_mlp_taktyka = struktura()

    epochs = 1000
    print(f"Rozpoczynam rygorystyczny trening {epochs} epok dla precyzyjnych wykresów...")

    for warstwa in ['warstwa_0', 'warstwa_1', 'warstwa_2']:
        print(f"\nPrzetwarzanie {warstwa.upper()}...")

        sonda_lin, sonda_mlp = trenuj_sondy(
            aktywacje_warstwy=aktywacje[warstwa],
            relatywne_trening=relatywne_trening,
            liczba_trening=liczba_trening,
            epochs=epochs
        )

        mysli_test = aktywacje[warstwa][liczba_trening:liczba_trening + liczba_test].view(-1, 128)

        with torch.no_grad():
            pred_lin = sonda_lin(mysli_test).view(-1, 3, 9)
            wyroki_lin = torch.argmax(pred_lin, dim=1)
            lin_wyniki.append(round((wyroki_lin == relatywne_test).float().mean().item() * 100, 1))

            pred_mlp = sonda_mlp(mysli_test).view(-1, 3, 9)
            wyroki_mlp = torch.argmax(pred_mlp, dim=1)
            mlp_wyniki.append(round((wyroki_mlp == relatywne_test).float().mean().item() * 100, 1))

            fizyka_radar = ((wyroki_mlp == 0) == (relatywne_test == 0)).float().mean().item() * 100
            fizyka_dane_mlp.append(round(fizyka_radar, 1))
            taktyka_dane_mlp.append(round((wyroki_mlp == relatywne_test).float().mean().item() * 100, 1))

            wyroki_lin_czas = wyroki_lin.view(-1, 9, 9)
            wyroki_mlp_czas = wyroki_mlp.view(-1, 9, 9)
            prawda_czas = relatywne_test.view(-1, 9, 9)

            # Nadal zbieramy dane dla L od 5 do 9, ale funkcja wykresu wyciągnie tylko L=9
            for L in range(5, 10):
                maska = (dlugosci_test == L)
                if maska.sum() == 0: continue

                wyr_lin_L = wyroki_lin_czas[maska]
                wyr_mlp_L = wyroki_mlp_czas[maska]
                prawda_L = prawda_czas[maska]

                for k in range(L):
                    prawdziwe_puste = (prawda_L[:, k, :] == 0)

                    # 1. Liniowa - Fizyka
                    zgadniete_puste_lin = (wyr_lin_L[:, k, :] == 0)
                    acc = (zgadniete_puste_lin == prawdziwe_puste).float().mean().item() * 100
                    dane_lin_fizyka[L][warstwa].append(round(acc, 1))

                    # 2. Liniowa - Taktyka
                    acc = (wyr_lin_L[:, k, :] == prawda_L[:, k, :]).float().mean().item() * 100
                    dane_lin_taktyka[L][warstwa].append(round(acc, 1))

                    # 3. MLP - Fizyka
                    zgadniete_puste_mlp = (wyr_mlp_L[:, k, :] == 0)
                    acc = (zgadniete_puste_mlp == prawdziwe_puste).float().mean().item() * 100
                    dane_mlp_fizyka[L][warstwa].append(round(acc, 1))

                    # 4. MLP - Taktyka
                    acc = (wyr_mlp_L[:, k, :] == prawda_L[:, k, :]).float().mean().item() * 100
                    dane_mlp_taktyka[L][warstwa].append(round(acc, 1))

    print("\nGenerowanie i zapisywanie skonsolidowanych raportów...")

    zapisz_wykres(plot_probe_comparison(lin_wyniki, mlp_wyniki), "01_zatarcie_liniowe.png")
    zapisz_wykres(plot_world_model_resolution(fizyka_dane_mlp, taktyka_dane_mlp), "02_obraz_swiata.png")

    # Tworzymy jeden zunifikowany obraz w siatce 2x2 z gier o długości L=9
    grid_fig = plot_attention_shift_L9_grid(dane_lin_fizyka, dane_mlp_fizyka, dane_lin_taktyka, dane_mlp_taktyka)
    zapisz_wykres(grid_fig, "03_zatarcie_czasowe_L9_grid.png")

    print("Zakończono! Jeden główny wykres zatarcia zapisano jako '../plots/03_zatarcie_czasowe_L9_grid.png'.")