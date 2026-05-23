import torch
import torch.nn as nn

# --- DEFINICJE SOND ---

class LinearProbe(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layer = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.layer(x)


class MLPProbe(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        return self.layer(x)


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