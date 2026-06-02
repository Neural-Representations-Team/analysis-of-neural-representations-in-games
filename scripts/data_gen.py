import random
import json
import os
import math
max_depth = 4
# --- LOGIKA GRY KÓŁKO I KRZYŻYK ---

def check_winner(board):
    """Sprawdza, czy ktoś wygrał. Plansza to lista 9 elementów (0: puste, 1: X, -1: O)."""
    win_conditions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], # Wiersze
        [0, 3, 6], [1, 4, 7], [2, 5, 8], # Kolumny
        [0, 4, 8], [2, 4, 6]             # Przekątne
    ]
    for condition in win_conditions:
        a, b, c = condition
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a] # Zwraca 1 (X) lub -1 (O)
            
    if 0 not in board:
        return 'Tie' # Remis
    return None # Gra toczy się dalej

def minimax(board, depth, is_maximizing):
    """Klasyczny algorytm Minimax do oceny stanu planszy."""
    winner = check_winner(board)
    # Zwracamy punktację zależną od głębokości (szybsza wygrana jest lepsza)
    if depth > max_depth: return 0
    if winner == 1: return 10 - depth
    if winner == -1: return -10 + depth
    if winner == 'Tie': return 0

    if is_maximizing:
        best_score = -math.inf
        for i in range(9):
            if board[i] == 0:
                board[i] = 1
                score = minimax(board, depth + 1, False)
                board[i] = 0
                best_score = max(score, best_score)
        return best_score
    else:
        best_score = math.inf
        for i in range(9):
            if board[i] == 0:
                board[i] = -1
                score = minimax(board, depth + 1, True)
                board[i] = 0
                best_score = min(score, best_score)
        return best_score

def get_optimal_moves(board, player):
    """Zwraca listę najlepszych możliwych ruchów dla danego gracza."""
    best_score = -math.inf if player == 1 else math.inf
    best_moves = []
    is_maximizing = (player == 1)

    for i in range(9):
        if board[i] == 0:
            board[i] = player
            score = minimax(board, 0, not is_maximizing)
            board[i] = 0

            # Zbieramy wszystkie ruchy, które dają optymalny wynik
            if is_maximizing:
                if score > best_score:
                    best_score = score
                    best_moves = [i]
                elif score == best_score:
                    best_moves.append(i)
            else:
                if score < best_score:
                    best_score = score
                    best_moves = [i]
                elif score == best_score:
                    best_moves.append(i)
                    
    return best_moves

# --- GENEROWANIE DANYCH ---

def generate_single_game(strategy="random"):
    """Generuje pojedynczą grę i zwraca sekwencję ruchów zakończoną tokenem 9."""
    board = [0] * 9
    moves = []
    current_player = 1 # 1 to X (zaczyna), -1 to O

    while True:
        winner = check_winner(board)
        if winner is not None:
            break

        if strategy == "random":
            available_moves = [i for i, val in enumerate(board) if val == 0]
            move = random.choice(available_moves)
        elif strategy == "minimax":
            optimal_moves = get_optimal_moves(board, current_player)
            # Losujemy spośród równorzędnych optymalnych ruchów, by dane były różnorodne
            move = random.choice(optimal_moves) 
        else:
            raise ValueError("Nieznana strategia. Wybierz 'random' lub 'minimax'.")

        board[move] = current_player
        moves.append(move)
        current_player *= -1

    moves.append(9) # Nasz specjalny token końca gry <END>
    return moves

def generate_dataset(num_games, strategy="random", seed=None, filename="dataset.json"):
    """
    Główna funkcja wywołująca.
    Generuje zbiór danych i zapisuje go do folderu /data/
    """
    if seed is not None:
        random.seed(seed)

    dataset = []
    for _ in range(num_games):
        print(f"Generowanie gry {_ + 1}/{num_games} (strategia: {strategy})...")
        dataset.append(generate_single_game(strategy))

    # Ustalanie ścieżki do folderu /data/ (jeden poziom wyżej niż /scripts/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    
    file_path = os.path.join(data_dir, filename)

    # Zapis do pliku JSON
    with open(file_path, "w") as f:
        json.dump(dataset, f)

    print(f"✅ Sukces! Wygenerowano {num_games} gier (strategia: {strategy}).")
    print(f"💾 Zapisano w: {file_path}")

if __name__ == "__main__":
    # Przykładowe wywołanie - możesz to dowolnie modyfikować
    print("Rozpoczynam generowanie danych...")
    
    # Generujemy 10 000 gier losowych do nauki podstawowych zasad
    generate_dataset(num_games=10000, strategy="random", seed=random.randint(0, 10000), filename="games.json")
    