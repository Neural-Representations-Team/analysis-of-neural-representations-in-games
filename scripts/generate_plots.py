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
    fig.savefig(sciezka, dpi=300, bbox_inches='tight')
    print(f"Zapisano: {sciezka}")
    plt.close(fig)  # Zamyka okno, aby nie zaśmiecać RAMu


# --- FUNKCJE RYSUJĄCE POZOSTAJĄ BEZ ZMIAN ---

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


def plot_attention_shift_by_length_and_probe(dane, tytul_glowny, ylabel):
    kolory = {'warstwa_0': '#8e44ad', 'warstwa_1': '#2c3e50', 'warstwa_2': '#e67e22'}
    nazwy = {'warstwa_0': 'W0 (Wejście)', 'warstwa_1': 'W1 (Kartograf)', 'warstwa_2': 'W2 (Decyzyjna)'}
    fig, axes = plt.subplots(5, 1, figsize=(10, 16), constrained_layout=True)
    fig.suptitle(tytul_glowny, fontsize=16, fontweight='bold')
    for i, L in enumerate(range(5, 10)):
        ax = axes[i]
        if not dane[L]['warstwa_0']: continue
        kroki = np.arange(1, L + 1)
        for w in ['warstwa_0', 'warstwa_1', 'warstwa_2']:
            ax.plot(kroki, dane[L][w], marker='o', linewidth=2, label=nazwy[w], color=kolory[w])
        ax.set_title(f'Gry kończące się w {L}. ruchu')
        ax.set_xticks(kroki)
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_ylim(30, 105)
        ax.grid(True, linestyle='--', alpha=0.7)
        if i == 0: ax.legend()
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

    # Pojemniki na ogólne statystyki
    lin_wyniki = []
    mlp_wyniki = []
    fizyka_dane_mlp = []
    taktyka_dane_mlp = []

    # Rozbudowane słowniki na 4 kategorie wykresów
    struktura = lambda: {L: {w: [] for w in ['warstwa_0', 'warstwa_1', 'warstwa_2']} for L in range(5, 10)}
    dane_lin_fizyka = struktura()
    dane_lin_taktyka = struktura()
    dane_mlp_fizyka = struktura()
    dane_mlp_taktyka = struktura()

    epochs = 1000
    print(f"Rozpoczynam rygorystyczny trening {epochs} epok dla precyzyjnych wykresów...")

    for warstwa in ['warstwa_0', 'warstwa_1', 'warstwa_2']:
        print(f"\nPrzetwarzanie {warstwa.upper()}...")

        # Błyskawiczny trening z wykorzystaniem utils
        sonda_lin, sonda_mlp = trenuj_sondy(
            aktywacje_warstwy=aktywacje[warstwa],
            relatywne_trening=relatywne_trening,
            liczba_trening=liczba_trening,
            epochs=epochs
        )

        # Wyciągamy myśli egzaminacyjne
        mysli_test = aktywacje[warstwa][liczba_trening:liczba_trening + liczba_test].view(-1, 128)

        with torch.no_grad():
            # Wyniki dla sondy Liniowej
            pred_lin = sonda_lin(mysli_test).view(-1, 3, 9)
            wyroki_lin = torch.argmax(pred_lin, dim=1)
            lin_wyniki.append(round((wyroki_lin == relatywne_test).float().mean().item() * 100, 1))

            # Wyniki dla sondy MLP
            pred_mlp = sonda_mlp(mysli_test).view(-1, 3, 9)
            wyroki_mlp = torch.argmax(pred_mlp, dim=1)
            mlp_wyniki.append(round((wyroki_mlp == relatywne_test).float().mean().item() * 100, 1))

            # Słupki (bierzemy tylko z MLP)
            fizyka_radar = ((wyroki_mlp == 0) == (relatywne_test == 0)).float().mean().item() * 100
            fizyka_dane_mlp.append(round(fizyka_radar, 1))
            taktyka_dane_mlp.append(round((wyroki_mlp == relatywne_test).float().mean().item() * 100, 1))

            # --- ANALIZA W CZASIE (DLA OBU SOND) ---
            wyroki_lin_czas = wyroki_lin.view(-1, 9, 9)
            wyroki_mlp_czas = wyroki_mlp.view(-1, 9, 9)
            prawda_czas = relatywne_test.view(-1, 9, 9)

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

    print("\nGenerowanie i zapisywanie raportów...")

    zapisz_wykres(plot_probe_comparison(lin_wyniki, mlp_wyniki), "01_zatarcie_liniowe.png")
    zapisz_wykres(plot_world_model_resolution(fizyka_dane_mlp, taktyka_dane_mlp), "02_obraz_swiata.png")
    zapisz_wykres(plot_attention_shift_by_length_and_probe(dane_lin_fizyka, 'Fizyka - SONDA LINIOWA', 'Poprawność (%)'),
                  "03_lin_fizyka.png")
    zapisz_wykres(
        plot_attention_shift_by_length_and_probe(dane_lin_taktyka, 'Taktyka - SONDA LINIOWA', 'Poprawność (%)'),
        "04_lin_taktyka.png")
    zapisz_wykres(plot_attention_shift_by_length_and_probe(dane_mlp_fizyka, 'Fizyka - SONDA MLP', 'Poprawność (%)'),
                  "05_mlp_fizyka.png")
    zapisz_wykres(plot_attention_shift_by_length_and_probe(dane_mlp_taktyka, 'Taktyka - SONDA MLP', 'Poprawność (%)'),
                  "06_mlp_taktyka.png")

    print("Wszystkie wykresy zapisane w folderze '../plots/'.")