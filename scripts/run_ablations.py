import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json

# --- EXTERNAL TOOLKIT IMPORTS ---
# Aliased to maintain strict compatibility with local parameters
from probe_utils import przygotuj_dane as prepare_data, trenuj_sondy as train_probes

plt.style.use('ggplot')


# --- MODEL DEFINITION ---
class TinyTicTacToeGPT(nn.Module):
    def __init__(self, d_model=128, num_layers=3, nhead=8):
        super().__init__()
        self.embedding = nn.Embedding(11, d_model)
        self.pos_encoder = nn.Embedding(10, d_model)
        decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 11)

    def forward(self, x, mask=None):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_encoder(positions)
        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        out = self.transformer(x, mask=mask)
        return self.fc_out(out)


# --- DATA PREPARATION ---
def load_test_games(start=1000, end=1200):
    """
    Extracts original game sequences directly from the JSON file
    to ensure tensor dimensions strictly align with the model's input (Batch x 9).
    """
    with open('../data/games.json', 'r') as file:
        game_data = json.load(file)

    sequences = []
    for moves in game_data[start:end]:
        sequence = moves[:9]
        # Pad with the End-of-Game token (9) to maintain a fixed sequence length of 9
        while len(sequence) < 9:
            sequence.append(9)
        sequences.append(sequence)

    return torch.tensor(sequences, dtype=torch.long)


# --- CORE EXPERIMENT LOGIC ---
def perform_ablation_study():
    print("Loading data...")
    # Kwargs kept identical to avoid TypeError with external script
    clean_activations, relative_train, relative_test, _ = prepare_data(
        sciezka_do_danych='../data/processed/dataset_pelny.pt',
        liczba_trening=1000,
        liczba_test=200
    )

    print("Step 1: Train the reference probe (Baseline Model)...")
    # Train the probe on the unablated Layer 1 spatial representations
    reference_probe, _ = train_probes(clean_activations['warstwa_1'], relative_train, 1000, epochs=500)

    clean_test_activations = clean_activations['warstwa_1'][1000:1200].view(-1, 128)

    with torch.no_grad():
        clean_preds = reference_probe(clean_test_activations).view(-1, 3, 9)
        clean_predictions = torch.argmax(clean_preds, dim=1)
        true_empty = (relative_test == 0)
        predicted_empty_clean = (clean_predictions == 0)
        baseline_spatial_acc = (predicted_empty_clean == true_empty).float().mean().item() * 100

    print(f"Baseline spatial representation accuracy: {baseline_spatial_acc:.1f}%\n")

    print("Step 2: Load the model and register extraction hooks...")
    model = TinyTicTacToeGPT()
    model.load_state_dict(
        torch.load('../models/transformer/tictactoe_model.pth', map_location='cpu', weights_only=True))
    model.eval()

    # Utilize standardized, one-dimensional input sequences
    test_games = load_test_games(1000, 1200)

    extracted_activations = {}

    def extraction_hook(module, input, output):
        extracted_activations['val'] = output.detach()

    model.transformer.layers[1].register_forward_hook(extraction_hook)

    ablation_results = np.zeros((3, 8))

    print("Step 3: Perform weight ablation...")

    d_model = 128
    nhead = 8
    head_dim = d_model // nhead

    for layer in range(3):
        for head in range(8):
            # Store original weights in memory
            original_weights = model.transformer.layers[layer].self_attn.out_proj.weight.data.clone()

            start_idx = head * head_dim
            end_idx = start_idx + head_dim

            # PHYSICAL ABLATION: Zero out the columns corresponding to the target attention head
            model.transformer.layers[layer].self_attn.out_proj.weight.data[:, start_idx:end_idx] = 0.0

            with torch.no_grad():
                # Process the standardized inputs
                _ = model(test_games)
                ablated_activations = extracted_activations['val'].view(-1, 128)

                ablated_preds = reference_probe(ablated_activations).view(-1, 3, 9)
                ablated_predictions = torch.argmax(ablated_preds, dim=1)

                predicted_empty_ablated = (ablated_predictions == 0)
                ablated_acc = (predicted_empty_ablated == true_empty).float().mean().item() * 100

            performance_drop = baseline_spatial_acc - ablated_acc
            ablation_results[layer, head] = performance_drop

            # Restore original weights before the next iteration
            model.transformer.layers[layer].self_attn.out_proj.weight.data = original_weights

    print("\nGenerating circuit map...")
    os.makedirs('../plots/ablations', exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)

    sns.heatmap(ablation_results, annot=True, fmt=".1f", cmap="Reds",
                xticklabels=[f"H{i}" for i in range(8)],
                yticklabels=[f"L{i}" for i in range(3)], ax=ax,
                cbar_kws={'label': 'Spatial Accuracy Drop (%)'})

    ax.set_title("Spatial Map Circuit Localization (Head Ablation)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Attention Head", fontweight='bold')
    ax.set_ylabel("Network Layer", fontweight='bold')

    plot_path = '../plots/ablations/07_spatial_circuit.png'
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Circuit localization proof saved to: {plot_path}")


if __name__ == '__main__':
    perform_ablation_study()