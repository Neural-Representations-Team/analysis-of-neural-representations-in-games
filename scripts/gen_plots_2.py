import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# --- UTILITIES ---
from probe_utils import przygotuj_dane, trenuj_sondy

plt.style.use('ggplot')


def save_plot(fig, filename):
    if not os.path.exists('../plots'):
        os.makedirs('../plots')
    filepath = os.path.join('../plots', filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close(fig)


def plot_probe_comparison(linear_acc, mlp_acc):
    layers = ['Layer 0', 'Layer 1', 'Layer 2']
    x = np.arange(len(layers))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.bar(x - width / 2, linear_acc, width, label='Linear Probe', color='#3498db')
    ax.bar(x + width / 2, mlp_acc, width, label='MLP Probe', color='#e74c3c')
    ax.set_ylabel('Test Accuracy (%)', fontweight='bold')
    fig.suptitle('Linear Probe Performance: Piece Color Detection', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_ylim(50, 100)
    ax.legend()
    for i, v in enumerate(linear_acc): ax.text(i - width / 2, v + 0.5, f"{v}%", ha='center')
    for i, v in enumerate(mlp_acc): ax.text(i + width / 2, v + 0.5, f"{v}%", ha='center')
    return fig


def plot_world_model_resolution(physics_data, tactics_data):
    layers = ['Layer 0', 'Layer 1', 'Layer 2']
    x = np.arange(len(layers))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.bar(x - width / 2, physics_data, width, label='Board Dynamics (Empty vs. Occupied)', color='#2ecc71')
    ax.bar(x + width / 2, tactics_data, width, label='Piece Ownership (Mine vs. Yours)', color='#9b59b6')
    ax.set_ylabel('Test Accuracy (%)', fontweight='bold')
    fig.suptitle('World Model Accuracy (MLP Probe)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_ylim(50, 105)
    ax.legend(loc='lower right')
    for i, v in enumerate(physics_data): ax.text(i - width / 2, v + 0.5, f"{v}%", ha='center')
    for i, v in enumerate(tactics_data): ax.text(i + width / 2, v + 0.5, f"{v}%", ha='center')
    return fig


def plot_representation_decay_l9_grid(lin_physics, mlp_physics, lin_tactics, mlp_tactics):
    colors = {
        'warstwa_0': '#e74c3c',
        'warstwa_1': '#2c3e50',
        'warstwa_2': '#3498db'
    }
    names = {
        'warstwa_0': 'L0 (Initial)',
        'warstwa_1': 'L1 (Spatial)',
        'warstwa_2': 'L2 (Decision)'
    }
    L = 9
    steps = np.arange(1, L + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    fig.suptitle('Representation Decay Over Time for Complete Games (9 moves)', fontsize=16, fontweight='bold',
                 y=0.95)

    configs = [
        (axes[0, 0], lin_physics, 'Board Dynamics (Empty vs. Occupied) - LINEAR PROBE'),
        (axes[0, 1], mlp_physics, 'Board Dynamics (Empty vs. Occupied) - MLP PROBE'),
        (axes[1, 0], lin_tactics, 'Piece Ownership (Mine vs. Yours) - LINEAR PROBE'),
        (axes[1, 1], mlp_tactics, 'Piece Ownership (Mine vs. Yours) - MLP PROBE')
    ]

    for ax, data_dict, title in configs:
        for layer in ['warstwa_0', 'warstwa_1', 'warstwa_2']:
            if not data_dict[L][layer]:
                continue
            ax.plot(steps, data_dict[L][layer], marker='o', markersize=6, linewidth=2.5,
                    label=names[layer], color=colors[layer])

        ax.set_title(title, fontsize=12)
        ax.set_ylim(30, 105)
        ax.set_xticks(steps)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for ax in axes[:, 0]:
        ax.set_ylabel('Decoding Accuracy (%)', fontweight='bold', fontsize=11)
    for ax in axes[1, :]:
        ax.set_xlabel('Move Number', fontweight='bold', fontsize=11)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.02), fontsize=12)

    plt.subplots_adjust(bottom=0.12, hspace=0.15, wspace=0.1)

    return fig


# --- MAIN PIPELINE ---

if __name__ == "__main__":
    n_train = 1000
    n_test = 200

    print("Loading data via utility functions...")
    activations, rel_train, rel_test, test_lengths = przygotuj_dane(
        sciezka_do_danych='../data/processed/dataset_pelny.pt',
        liczba_trening=n_train,
        liczba_test=n_test
    )

    lin_results = []
    mlp_results = []
    physics_data_mlp = []
    tactics_data_mlp = []

    data_structure = lambda: {L: {w: [] for w in ['warstwa_0', 'warstwa_1', 'warstwa_2']} for L in range(5, 10)}
    data_lin_physics = data_structure()
    data_lin_tactics = data_structure()
    data_mlp_physics = data_structure()
    data_mlp_tactics = data_structure()

    epochs = 1000
    print(f"Initiating probe training for {epochs} epochs...")

    for layer in ['warstwa_0', 'warstwa_1', 'warstwa_2']:
        print(f"\nProcessing {layer.upper()}...")

        lin_probe, mlp_probe = trenuj_sondy(
            aktywacje_warstwy=activations[layer],
            relatywne_trening=rel_train,
            liczba_trening=n_train,
            epochs=epochs
        )

        test_thoughts = activations[layer][n_train:n_train + n_test].view(-1, 128)

        with torch.no_grad():
            pred_lin = lin_probe(test_thoughts).view(-1, 3, 9)
            preds_lin_idx = torch.argmax(pred_lin, dim=1)
            lin_results.append(round((preds_lin_idx == rel_test).float().mean().item() * 100, 1))

            pred_mlp = mlp_probe(test_thoughts).view(-1, 3, 9)
            preds_mlp_idx = torch.argmax(pred_mlp, dim=1)
            mlp_results.append(round((preds_mlp_idx == rel_test).float().mean().item() * 100, 1))

            physics_radar = ((preds_mlp_idx == 0) == (rel_test == 0)).float().mean().item() * 100
            physics_data_mlp.append(round(physics_radar, 1))
            tactics_data_mlp.append(round((preds_mlp_idx == rel_test).float().mean().item() * 100, 1))

            preds_lin_time = preds_lin_idx.view(-1, 9, 9)
            preds_mlp_time = preds_mlp_idx.view(-1, 9, 9)
            ground_truth_time = rel_test.view(-1, 9, 9)

            for L in range(5, 10):
                mask = (test_lengths == L)
                if mask.sum() == 0: continue

                pred_lin_l = preds_lin_time[mask]
                pred_mlp_l = preds_mlp_time[mask]
                ground_truth_l = ground_truth_time[mask]

                for k in range(L):
                    true_empty = (ground_truth_l[:, k, :] == 0)

                    # 1. Linear Probe - Board Dynamics
                    pred_empty_lin = (pred_lin_l[:, k, :] == 0)
                    acc = (pred_empty_lin == true_empty).float().mean().item() * 100
                    data_lin_physics[L][layer].append(round(acc, 1))

                    # 2. Linear Probe - Tactics
                    acc = (pred_lin_l[:, k, :] == ground_truth_l[:, k, :]).float().mean().item() * 100
                    data_lin_tactics[L][layer].append(round(acc, 1))

                    # 3. MLP Probe - Board Dynamics
                    pred_empty_mlp = (pred_mlp_l[:, k, :] == 0)
                    acc = (pred_empty_mlp == true_empty).float().mean().item() * 100
                    data_mlp_physics[L][layer].append(round(acc, 1))

                    # 4. MLP Probe - Tactics
                    acc = (pred_mlp_l[:, k, :] == ground_truth_l[:, k, :]).float().mean().item() * 100
                    data_mlp_tactics[L][layer].append(round(acc, 1))

    print("\nGenerating and saving consolidated reports...")

    save_plot(plot_probe_comparison(lin_results, mlp_results), "01_linear_probe_performance.png")
    save_plot(plot_world_model_resolution(physics_data_mlp, tactics_data_mlp), "02_world_model_accuracy.png")

    grid_fig = plot_representation_decay_l9_grid(data_lin_physics, data_mlp_physics, data_lin_tactics, data_mlp_tactics)
    save_plot(grid_fig, "03_representation_decay_l9_grid.png")

    print("Completed. Main representation decay plot saved as '../plots/03_representation_decay_l9_grid.png'.")