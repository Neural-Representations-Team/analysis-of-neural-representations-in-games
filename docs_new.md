### Spis Treści

**1. Wstęp**

* 1.1. Cel Projektu (Mechanistyczna Interpretowalność)
* 1.2. Teza Badawcza (Wyłaniające się Reprezentacje Świata)
* 1.3. Krótkie Podsumowanie Wyników

**2. Metodologia i Architektura Modelu**

* 2.1. Charakterystyka Modelu
* 2.2. Zbiór Danych
* *Opis środowiska treningowego: wyłącznie losowe, legalne ruchy.*
* 2.3. Funkcja Celu (Next-Token Prediction & Legalność)

**3. Ekstrakcja Aktywacji**

* 3.1. Mechanizm Forward Hooks
* 3.2. Reprezentacja Danych

**4. Sondowanie Liniowe (Linear Probing) - Odkrywanie Mapy Zjawisk**

* 4.1. Definicja i Trening Sond (Liniowa vs MLP)
* 4.2. Wyniki Detekcji: Fizyka Planszy (Puste vs Zajęte)
* 4.3. Wyniki Detekcji: Taktyka (Mój Pionek vs Pionek Wroga)
* 4.4. Zjawisko Zatarcia Liniowego (Attention Shift w Czasie)

**5. Interwencje Przyczynowe (Causal Interventions) - Hakowanie Modelu**

* 5.1. Definicja Dowodu Przyczynowo-Skutkowego
* 5.2. Wymuszanie Decyzji (Modyfikacja Wektora "Pole Puste" na "Pole Zajęte")
* 5.3. Efekt Fali Uderzeniowej (Kalibracja Siły Interwencji)
* 5.4. Asymetria Przyczynowa i Samonaprawa (Self-Healing) - Analiza Warstwa po Warstwie

**6. Podsumowanie i Wnioski**

* 6.1. Weryfikacja Tezy Głównej
* 6.2. Odporność Architektury Transformer (Redundancja i Obwody Zapasowe)
* 6.3. Znaczenie Projektu dla Dziedziny Interpretowalności AI

Zrozumiałem. Przygotowałem dla Ciebie ten fragment w profesjonalnym, akademickim stylu, idealnym do wklejenia prosto do pliku `README.md`. Użyłem jasnego formatowania i pogrubień, aby tekst był czytelny i łatwy do przyswojenia zarówno dla Ciebie, jak i dla komisji.

Oto Twój pierwszy rozdział:

---

# 1. Wstęp

Zrozumienie, w jaki sposób sztuczne sieci neuronowe przetwarzają informacje i podejmują decyzje, jest obecnie jednym z największych wyzwań w dziedzinie sztucznej inteligencji. Choć modele językowe osiągają niesamowite wyniki, ich wewnętrzne mechanizmy często pozostają nieprzeniknioną "czarną skrzynką". Ten projekt ma na celu rzucenie światła na te ukryte procesy.

### 1.1. Cel Projektu (Mechanistyczna Interpretowalność)

Głównym celem pracy jest przeprowadzenie **Mechanistic Interpretability** [Mechanistycznej Interpretowalności - dziedziny nauki polegającej na inżynierii odwrotnej sieci neuronowych w celu zdekodowania ich wewnętrznych algorytmów i procesów decyzyjnych].

Jako środowisko testowe wybrano grę w Kółko i Krzyżyk (Tic-Tac-Toe). Zbudowano i wytrenowano autoregresyjny model językowy oparty na architekturze Transformer, uczony wyłącznie przewidywania kolejnego prawidłowego ruchu na podstawie płaskiej sekwencji znaków. Projekt skupia się na stworzeniu narzędzi analitycznych, które pozwolą "zajrzeć do wnętrza" tego modelu i zrozumieć, w jaki sposób analizuje on grę bez dostępu do wizualnego obrazu planszy.

### 1.2. Teza Badawcza (Wyłaniające się Reprezentacje Świata)

Punktem wyjścia dla naszych badań jest hipoteza o istnieniu **Emergent World Representations** [Wyłaniających się Reprezentacji Świata - zjawiska, w którym sieć uczona na jednowymiarowym tekście samoistnie buduje wewnętrzny, wielowymiarowy model fizycznej rzeczywistości, w której operuje].

**Teza projektu brzmi:** Autoregresyjny model Transformer, trenowany wyłącznie na losowych, legalnych sekwencjach ruchów (bez implementacji strategii wygrywającej), nie zapamiętuje jedynie statystycznych ciągów cyfr. Aby skutecznie unikać generowania nielegalnych ruchów, model ten samoistnie wykształca w swoich ukrytych warstwach zgeometryzowaną, przestrzenną mapę planszy oraz aktywnie z niej korzysta podczas podejmowania decyzji.

### 1.3. Krótkie Podsumowanie Wyników

Przeprowadzone eksperymenty dostarczyły twardych dowodów potwierdzających postawioną tezę, ujawniając jednocześnie złożoność wewnętrznej reprezentacji modelu. Najważniejsze wnioski to:

* **Zdekodowanie Mapy Przestrzennej (Linear & Non-linear Probing):** Udowodniono, że model zbudował wewnątrz siebie rozwarstwioną reprezentację planszy. Prosty, fizyczny stan gry (czy pole jest puste, czy zajęte) jest idealnie i liniowo zakodowany w 128-wymiarowej przestrzeni ukrytej (co wykazano za pomocą sond liniowych). Z kolei głębsza, taktyczna świadomość (odróżnienie własnego pionka od pionka przeciwnika) wymagała zastosowania nieliniowych sond (sieci MLP), co dowodzi większej złożoności tego konceptu wewnątrz wag sieci.
* **Asymetria Dowodu Przyczynowo-Skutkowego (Causal Proof):** Przeprowadzono Interwencje Przyczynowe [Celowe wprowadzanie błędnych danych do warstw działającej sieci], modyfikując pamięć modelu w czasie rzeczywistym. Wykazano ewidentną asymetrię w jego zachowaniu:
* Zmiana informacji z "pole puste" na "pole zajęte" zawsze skutecznie blokowała możliwość ruchu w to miejsce, niezależnie od tego, w której warstwie dokonano interwencji.
* Zmiana informacji z "pole zajęte" na "pole puste" nie przynosiła efektu we wczesnych fazach przetwarzania. Model ignorował sztuczne pozwolenie i nadal odmawiał wykonania ruchu na zajęte pole.


* **Odkrycie Obwodów Zapasowych (Backup Circuits):** Zjawisko ignorowania pozwolenia na ruch stanowi dowód na istnienie potężnych mechanizmów obronnych wewnątrz sieci. Udowodniono, że we wczesnych warstwach (L0, L1) głowy uwagi (Self-Attention) aktywnie weryfikują zmanipulowaną mapę przestrzenną z fizyczną historią ruchów podaną na wejściu. Widząc, że dane pole zostało już wcześniej zagrane, sieć koryguje "kłamstwo" badacza i twardo blokuje ruch. Oszukanie modelu powiodło się dopiero w ostatniej warstwie (L2), gdzie mechanizmy uwagi już nie operują. Dowodzi to silnej redundancji [Istnienie obwodów zapasowych, które chronią system przed awarią] w architekturze Transformer.

---


Zrozumiałem. Bierzemy się za wielki finał dokumentacji. Złożymy te wnioski tak, aby były czytelne, miały mocne oparcie w naszych danych i utrzymywały pełen rygor naukowy.

Oto gotowy tekst do siódmego rozdziału. Zadbałem o przejrzystą strukturę i krótkie akapity, aby łatwo się to czytało.

---

# 6. Podsumowanie i Wnioski

### 6.1. Weryfikacja Tezy Głównej

Projekt jednoznacznie potwierdza postawioną tezę badawczą. Autoregresyjny model językowy, trenowany bez jakiejkolwiek strategii wygrywającej i wyłącznie na losowych sekwencjach poprawnych ruchów, nie zapamiętuje "w ciemno" statystycznych wzorców tekstu. Zamiast tego, w procesie optymalizacji i unikania błędów (kar za wygenerowanie ruchu w niedozwolone miejsce), model samoistnie wykształca **Emergent World Representations** [Wyłaniające się Reprezentacje Świata - wewnętrzny, zgeometryzowany model fizycznej rzeczywistości].

Zdekodowanie tej mapy za pomocą sond liniowych oraz weryfikacja jej działania za pomocą interwencji przyczynowych stanowi twardy dowód. Pokazuje on, że sieć przed wygenerowaniem kolejnego znaku aktywnie "rozumie" fizyczny układ planszy i opiera na nim swoje decyzje.

### 6.2. Odporność Architektury Transformer (Redundancja i Obwody Zapasowe)

Badania nad modyfikacją pamięci sieci ujawniły nieoczekiwaną dojrzałość i złożoność w tak małej architekturze. Próby sztucznego "pozwolenia" modelowi na zrobienie ruchu w zajęte już pole doprowadziły do odkrycia zjawiska **Self-Healing** [Samonaprawa Sieci - zdolność modelu do korygowania fałszywych informacji wewnątrz własnej pamięci].

Udowodniono, że wczesne warstwy sieci (Warstwa 0 i Warstwa 1) aktywnie współpracują z mechanizmem **Self-Attention** [Głowy Uwagi - moduły skanujące początkową historię ruchów podaną na wejściu]. Kiedy nasza zmanipulowana mapa przestrzenna sugerowała, że zajęte pole jest puste, głowy uwagi natychmiast weryfikowały to z twardą listą wcześniejszych ruchów i skutecznie blokowały ten błąd. Nasze oszustwo zadziałało dopiero w Warstwie 2 (na samym końcu modelu). To jest twardy dowód na to, że architektura Transformer naturalnie wykształca **Redundancję** [Istnienie obwodów zapasowych, które działają niezależnie, aby chronić system przed awarią i błędnymi decyzjami].

### 6.3. Znaczenie Projektu dla Dziedziny Interpretowalności AI

Wyniki tego projektu stanowią istotny, praktyczny wkład w rozwój dziedziny **Mechanistic Interpretability** [Mechanistyczna Interpretowalność - nauka o inżynierii odwrotnej i dekodowaniu czarnych skrzynek sztucznej inteligencji].

Praca ta dostarcza empirycznych dowodów na to, że nawet proste modele językowe realizujące zadanie przewidywania kolejnego znaku (Next-Token Prediction) są zdolne do budowania precyzyjnych i odpornych na manipulacje symulacji swojego środowiska. Zrozumienie, w jaki sposób modele te uczą się pojęć przestrzennych i jak ich wewnętrzne obwody same korygują błędy, jest kluczowym krokiem w stronę tworzenia bezpiecznej i przewidywalnej sztucznej inteligencji. Metodologia użyta w tej pracy, polegająca na łączeniu sond wektorowych z interwencjami przyczynowymi, z powodzeniem może zostać przeskalowana do analizy znacznie potężniejszych sieci neuronowych.

