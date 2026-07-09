import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os

# --- EXTERNAL TOOLKIT IMPORTS ---
# Aliased to maintain strict compatibility with your local files
from probe_utils import przygotuj_dane as prepare_data, trenuj_sondy as train_probes

plt.style.use('ggplot')


# --- 1. MODEL DEFINITION ---
class TinyTicTacToeGPT(nn.Module):
    def __init__(self, d_model=128, num_layers=3, nhead=8):
        super().__init__()
        self.embedding = nn.Embedding(11, d_model)
        self.pos_encoder = nn.Embedding(10, d_model)
        decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 11)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_encoder(positions)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        return self.transformer(x, mask=mask)


def train_and_extract_concept_vector(target_square, layer='warstwa_1', d_model=128):
    print(f"\n--- On-the-fly probing for square {target_square} at {layer} ---")

    # kwargs kept identical to ensure compatibility with probe_utils.py
    activations, relative_train, _, _ = prepare_data(
        sciezka_do_danych='../data/processed/dataset_pelny.pt',
        liczba_trening=500,
        liczba_test=10
    )

    linear_probe, _ = train_probes(
        aktywacje_warstwy=activations[layer],
        relatywne_trening=relative_train,
        liczba_trening=500,
        epochs=300
    )

    weights = next(linear_probe.parameters()).detach().cpu()

    # CORRECTED INDEX MATH (Aligned with .view(-1, 3, 9))
    idx_empty = 0 * 9 + target_square
    idx_X = 1 * 9 + target_square
    idx_O = 2 * 9 + target_square

    weight_empty = weights[idx_empty, :]
    weight_X = weights[idx_X, :]
    weight_O = weights[idx_O, :]

    # Occupancy Vector: Shift towards X and O, subtract Empty baseline
    concept_vector = ((weight_X + weight_O) / 2.0) - weight_empty

    # Vector normalization
    concept_vector = concept_vector / torch.norm(concept_vector)
    return concept_vector


def analyze_residual_stream(model_path, game_sequence, target_square):
    device = torch.device('cpu')

    concept_vector = train_and_extract_concept_vector(target_square, layer='warstwa_1')

    print("\n--- Residual Stream Analysis ---")
    model = TinyTicTacToeGPT(d_model=128, num_layers=3, nhead=8).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    x = torch.tensor([game_sequence], dtype=torch.long).to(device)

    stream_history = []
    hook_handles = []

    def pre_hook_l0(module, input_tensor):
        stream = input_tensor[0][0, -1, :]
        projection = torch.dot(stream, concept_vector).item()
        stream_history.append(("Start (Embeddings Only)", projection))

    def post_layer_hook(layer_name):
        def hook(module, input_tensor, output_tensor):
            stream = output_tensor[0, -1, :]
            projection = torch.dot(stream, concept_vector).item()
            stream_history.append((f"Post-Layer {layer_name}", projection))

        return hook

    hook_handles.append(model.transformer.layers[0].register_forward_pre_hook(pre_hook_l0))
    hook_handles.append(model.transformer.layers[0].register_forward_hook(post_layer_hook("L0")))
    hook_handles.append(model.transformer.layers[1].register_forward_hook(post_layer_hook("L1")))
    hook_handles.append(model.transformer.layers[2].register_forward_hook(post_layer_hook("L2 (Final)")))

    with torch.no_grad():
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=device).unsqueeze(0)
        input_emb = model.embedding(x) + model.pos_encoder(positions)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
        _ = model.transformer(input_emb, mask=mask)

    for handle in hook_handles:
        handle.remove()

    stages = [val[0] for val in stream_history]
    values = [val[1] for val in stream_history]

    plt.figure(figsize=(10, 6))

    # Positive signal representation
    plt.plot(stages, values, marker='o', linestyle='-', linewidth=3, markersize=10, color='#2ecc71')
    plt.fill_between(stages, values, color='#2ecc71', alpha=0.1)

    plt.title(
        f"Accumulation of Square Occupancy Representation in the Residual Stream\n(Target square: {target_square}, Game sequence: {game_sequence})",
        fontsize=14, fontweight='bold'
    )
    plt.ylabel("Signal Strength (Projection on 'Occupied' vector)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    for i, txt in enumerate(values):
        plt.annotate(f"{txt:.2f}", (stages[i], values[i]), textcoords="offset points", xytext=(0, 10), ha='center',
                     fontsize=12, fontweight='bold')

    print("\n--- RESIDUAL STREAM EVIDENCE RESULTS ---")
    for stage, val in stream_history:
        print(f"{stage}: {val:.3f}")

    plt.tight_layout()
    os.makedirs('../plots', exist_ok=True)
    plot_path = '../plots/residual_stream_proof.png'
    plt.savefig(plot_path, dpi=300)
    print(f"\nPlot saved to: {plot_path}")
    plt.show()


if __name__ == "__main__":
    MODEL_PATH = '../models/transformer/tictactoe_model.pth'

    # Execute a game sequence where Square 0 is occupied immediately
    test_game = [0, 4, 1, 8, 3]

    # Objective: Demonstrate the network's representation of the occupied target square
    analyze_residual_stream(MODEL_PATH, test_game, target_square=0)