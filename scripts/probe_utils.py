import torch
import torch.nn as nn

# --- DEFINICJE SOND ---

class MLPProbe(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 22),
            nn.ReLU(),
            nn.Linear(22, output_size)
        )

    def forward(self, x):
        return self.layer(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LinearProbe(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layer = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.layer(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# --- FUNKCJE POMOCNICZE ---

def tlumacz_na_relatywna(plansze_surowe):
    """
    Zamienia pionki graczy na logikę 'Mój pionek' (1) i 'Pionek Wroga' (2).
    """
    real_plane = plansze_surowe % 3
    liczba_jedynek = (real_plane == 1).sum(dim=1, keepdim=True)
    liczba_dwojek = (real_plane == 2).sum(dim=1, keepdim=True)
    ruch_dwojki = liczba_jedynek > liczba_dwojek
    relatywna_plansza = real_plane.clone()
    maska_ruchu = ruch_dwojki.expand_as(real_plane)
    relatywna_plansza[maska_ruchu & (real_plane == 1)] = 2
    relatywna_plansza[maska_ruchu & (real_plane == 2)] = 1
    return relatywna_plansza


def przygotuj_dane(sciezka_do_danych='data/processed/dataset_pelny.pt', liczba_trening=1000, liczba_test=200):
    """
    Ładuje dane z dysku, tłumaczy je i dzieli na pule: treningową oraz egzaminacyjną.
    Zwraca również precyzyjne długości gier testowych do wykresów.
    """
    dane_z_dysku = torch.load(sciezka_do_danych)
    aktywacje = dane_z_dysku['aktywacje']
    plansze = dane_z_dysku['plansze']

    plansze_trening = plansze[:liczba_trening].view(-1, 9)
    plansze_test = plansze[liczba_trening:liczba_trening + liczba_test].view(-1, 9)

    relatywne_trening = tlumacz_na_relatywna(plansze_trening)
    relatywne_test = tlumacz_na_relatywna(plansze_test)

    # Obliczanie faktycznej długości gier testowych (wymiar 3D)
    plansze_test_3d = plansze_test.view(-1, 9, 9)
    dlugosci_test = (plansze_test_3d[:, -1, :] != 0).sum(dim=1)
    dlugosci_test = torch.clamp(dlugosci_test, min=5, max=9)

    return aktywacje, relatywne_trening, relatywne_test, dlugosci_test


def trenuj_sondy(aktywacje_warstwy, relatywne_trening, liczba_trening=1000, epochs=1000):
    """
    Szybki silnik trenujący. Przyjmuje myśli sieci z danej warstwy i zwraca dwie w pełni gotowe sondy.
    """
    mysli_trening = aktywacje_warstwy[:liczba_trening].view(-1, 128)

    sonda_lin = LinearProbe(128, 27)
    opt_lin = torch.optim.Adam(sonda_lin.parameters(), lr=0.01)

    sonda_mlp = MLPProbe(128, 27)
    opt_mlp = torch.optim.Adam(sonda_mlp.parameters(), lr=0.005)

    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        # Trening Sondy Liniowej
        opt_lin.zero_grad()
        loss_lin = criterion(sonda_lin(mysli_trening).view(-1, 3, 9), relatywne_trening)
        loss_lin.backward()
        opt_lin.step()

        # Trening Sondy MLP
        opt_mlp.zero_grad()
        loss_mlp = criterion(sonda_mlp(mysli_trening).view(-1, 3, 9), relatywne_trening)
        loss_mlp.backward()
        opt_mlp.step()

    return sonda_lin, sonda_mlp


import plotly.graph_objects as go
from sklearn.decomposition import PCA
import numpy as np
import os

def wizualizuj_geometrie_pca_3d(wagi_sondy, tytul="Geometria Wag Sondy Liniowej w 3D"):
    """
    Tworzy interaktywny wykres 3D PCA pokazujący, jak model widzi geometrię planszy.
    Zapisuje wynik jako plik HTML, który można obracać w przeglądarce.
    """
    print(f"Generowanie interaktywnego wykresu 3D: {tytul}...")

    # 1. Pobieramy wagi dla 9 pól (stan 0 = Pole Puste)
    wektory = wagi_sondy[0:9, :].cpu().numpy()

    # 2. Redukcja wymiarów do 3D za pomocą PCA
    pca = PCA(n_components=3)
    wektory_3d = pca.fit_transform(wektory)

    # Rozdzielenie współrzędnych dla ułatwienia
    x = wektory_3d[:, 0]
    y = wektory_3d[:, 1]
    z = wektory_3d[:, 2]

    # 3. Definicja połączeń (siatka planszy)
    polaczenia = [
        (0, 1), (1, 2),  # Poziom: górny rząd
        (3, 4), (4, 5),  # Poziom: środkowy rząd
        (6, 7), (7, 8),  # Poziom: dolny rząd
        (0, 3), (3, 6),  # Pion: lewa kolumna
        (1, 4), (4, 7),  # Pion: środkowa kolumna
        (2, 5), (5, 8)   # Pion: prawa kolumna
    ]

    # 4. Budowanie linii (krawędzi łączących punkty)
    # Wstawiamy None między parami, żeby linie nie łączyły się w jedną długą nitkę
    edge_x, edge_y, edge_z = [], [], []
    for p1, p2 in polaczenia:
        edge_x.extend([x[p1], x[p2], None])
        edge_y.extend([y[p1], y[p2], None])
        edge_z.extend([z[p1], z[p2], None])

    trace_edges = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='gray', width=4),
        hoverinfo='none'
    )

    # 5. Budowanie punktów (węzłów - nasze 9 pól)
    trace_nodes = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers+text',
        marker=dict(size=8, color='red', line=dict(width=2, color='black')),
        text=[str(i) for i in range(9)], # Numeracja pól od 0 do 8
        textposition="top center",
        textfont=dict(size=16, color='black', family="Arial Black"),
        hoverinfo='text'
    )

    # 6. Składanie wykresu w jedną całość
    fig = go.Figure(data=[trace_edges, trace_nodes])

    fig.update_layout(
        title=tytul,
        showlegend=False,
        scene=dict(
            xaxis_title='Składowa 1 (PCA)',
            yaxis_title='Składowa 2 (PCA)',
            zaxis_title='Składowa 3 (PCA)'
        ),
        width=900,
        height=700,
        margin=dict(l=0, r=0, b=0, t=40) # Mniejsze marginesy
    )

    # 7. Zapis pliku jako interaktywny HTML
    os.makedirs('../plots/geometria', exist_ok=True)
    czysta_nazwa = tytul.replace(" ", "_").replace(":", "")
    sciezka = f"../plots/geometria/{czysta_nazwa}.html"

    fig.write_html(sciezka)
    print(f"Zapisano pomyślnie. Otwórz ten plik w przeglądarce: {sciezka}")