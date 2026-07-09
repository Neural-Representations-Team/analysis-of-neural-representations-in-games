import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# --- UTILITIES ---
from probe_utils import LinearProbe, MLPProbe, przygotuj_dane, wizualizuj_geometrie_pca_3d


def plot_learning_curves(train_losses, test_losses, train_accs, test_accs, epochs_x, layer_name, probe_type):
    """
    Generates and saves Loss and Accuracy plots to evaluate model convergence and detect potential overfitting.
    """
    os.makedirs('../plots/learning_curves', exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curve
    ax1.plot(epochs_x, train_losses, label='Train Loss', color='blue', alpha=0.7)
    ax1.plot(epochs_x, test_losses, label='Test Loss', color='red', alpha=0.7)
    ax1.set_title(f'Loss Curve - {probe_type} Probe ({layer_name})')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (CrossEntropy)')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Accuracy Curve
    ax2.plot(epochs_x, train_accs, label='Train Accuracy', color='blue', alpha=0.7)
    ax2.plot(epochs_x, test_accs, label='Test Accuracy', color='red', alpha=0.7)
    ax2.set_title(f'Accuracy Curve - {probe_type} Probe ({layer_name})')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    filepath = f'../plots/learning_curves/learning_curve_{layer_name}_{probe_type}.png'
    plt.savefig(filepath)
    plt.close()
    print(f"Saved learning curves: {filepath}")


# --- MAIN PIPELINE ---

print("Loading data via utility functions...")
activations, rel_train, rel_test, _ = przygotuj_dane(
    sciezka_do_danych='../data/processed/dataset_pelny.pt',
    liczba_trening=1000,
    liczba_test=200
)

n_train_games = 1000
n_test_games = 200

for layer_name in ['warstwa_0', 'warstwa_1', 'warstwa_2']:
    print(f"\n{'=' * 50}")
    print(f"INITIATING ANALYSIS FOR: {layer_name.upper()}")
    print(f"{'=' * 50}")

    layer_activations = activations[layer_name]

    # Segregating activations into train/test splits
    train_activations = layer_activations[:n_train_games].view(-1, 128)
    test_activations = layer_activations[n_train_games: n_train_games + n_test_games].view(-1, 128)

    for probe_type, probe_class in [('Linear', LinearProbe), ('Non-linear', MLPProbe)]:
        print(f"\n--- Training Probe: {probe_type} ---")

        # Probe Initialization
        probe = probe_class(128, 27)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(probe.parameters(), lr=0.005)

        epochs = 1000

        # Metrics storage
        hist_train_loss = []
        hist_test_loss = []
        hist_train_acc = []
        hist_test_acc = []
        saved_epochs = []

        for epoch in range(epochs):
            probe.train()
            optimizer.zero_grad()

            # Forward pass (Training Set)
            train_preds = probe(train_activations).view(-1, 3, 9)
            loss_train = criterion(train_preds, rel_train)

            loss_train.backward()
            optimizer.step()

            # Metric evaluation and logging
            if epoch % 10 == 0 or epoch == epochs - 1:
                probe.eval()
                with torch.no_grad():
                    # Training Metrics
                    chosen_states_train = torch.argmax(train_preds, dim=1)
                    correct_train = (chosen_states_train == rel_train).sum().item()
                    acc_train = (correct_train / (n_train_games * 9 * 9)) * 100

                    # Test Metrics
                    test_preds = probe(test_activations).view(-1, 3, 9)
                    loss_test = criterion(test_preds, rel_test)

                    chosen_states_test = torch.argmax(test_preds, dim=1)
                    correct_test = (chosen_states_test == rel_test).sum().item()
                    acc_test = (correct_test / (n_test_games * 9 * 9)) * 100

                    # Append to history
                    hist_train_loss.append(loss_train.item())
                    hist_test_loss.append(loss_test.item())
                    hist_train_acc.append(acc_train)
                    hist_test_acc.append(acc_test)
                    saved_epochs.append(epoch)

            # Console output
            if epoch % 100 == 0 or epoch == epochs - 1:
                print(
                    f"Epoch: {epoch + 1:4d}/{epochs} | Train Acc: {acc_train:.1f}% (Loss: {loss_train.item():.3f}) | TEST Acc: {acc_test:.1f}% (Loss: {loss_test.item():.3f})")

        # --- VISUALIZATION AFTER TRAINING ---
        plot_learning_curves(
            hist_train_loss, hist_test_loss,
            hist_train_acc, hist_test_acc,
            saved_epochs, layer_name, probe_type
        )

        # --- WEIGHT EXTRACTION FOR PCA ---
        if probe_type == 'Linear':
            final_weights = probe.layer.weight.data
        else:
            final_weights = probe.layer[2].weight.data

        wizualizuj_geometrie_pca_3d(final_weights, tytul=f"Network Geometry - {probe_type} Probe {layer_name}")

print("\n" + "=" * 50)
print("INITIATING BOARD STATE RECONSTRUCTION FOR THE FINAL LAYER (TEST SET)")
print("=" * 50)


def visualize_board_state(probe_model, test_acts, rel_test_data, game_idx=0):
    probe_model.eval()
    start = game_idx * 9
    end = start + 9

    game_activations = test_acts[start:end]
    ground_truth_board = rel_test_data[start:end]

    with torch.no_grad():
        predictions = probe_model(game_activations).view(9, 3, 9)
        probe_decisions = torch.argmax(predictions, dim=1)

    symbols = {0: "⬜", 1: "🟦", 2: "🟥"}

    for step in range(9):
        print(f"\n--- STEP {step + 1} ---")
        print("GROUND TRUTH        PROBE PREDICTION")
        true_step = ground_truth_board[step].tolist()
        probe_step = probe_decisions[step].tolist()

        for row in range(3):
            p_w = " ".join([symbols[true_step[row * 3 + i]] for i in range(3)])
            s_w = " ".join([symbols[probe_step[row * 3 + i]] for i in range(3)])
            print(f"{p_w}   |   {s_w}")


visualize_board_state(probe, test_activations, rel_test, game_idx=0)