import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
import os
import numpy as np

# --- EXTERNAL TOOLKIT IMPORTS ---
# Aliased to maintain structural compatibility with the original codebase.
from probe_utils import przygotuj_dane as prepare_data


# --- MODEL AND PROBE DEFINITION ---
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
        out = self.transformer(x, mask=mask)
        return self.fc_out(out)


class LinearProbe(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layer = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.layer(x)


# --- UTILITY FUNCTIONS ---
def get_padded_sequence(move_sequence):
    """Pads the sequence to a fixed length of 9 using the end-of-game token (9)."""
    seq = move_sequence[:9]
    while len(seq) < 9:
        seq.append(9)
    return seq


def evaluate_physics_accuracy(probe, activations, true_board_states):
    """Calculates spatial representation accuracy against true board states."""
    with torch.no_grad():
        predictions = probe(activations).view(-1, 3, 9)
        predicted_classes = torch.argmax(predictions, dim=1)
        # Physics validation: 0 (empty) vs occupied (1 or 2)
        physics_accuracy = ((predicted_classes == 0) == (true_board_states == 0)).float().mean().item() * 100
    return physics_accuracy


def ablate_attention_heads(model, layer_idx, head_list):
    """
    Performs physical ablation of specified attention heads by zeroing out
    their respective output projection weights.
    """
    d_model = 128
    nhead = 8
    head_dim = d_model // nhead

    with torch.no_grad():
        out_proj = model.transformer.layers[layer_idx].self_attn.out_proj
        for h in head_list:
            start_idx = h * head_dim
            end_idx = (h + 1) * head_dim
            # Zero out columns corresponding to the target head
            out_proj.weight[:, start_idx:end_idx] = 0.0


# --- CORE EXPERIMENT LOGIC ---
def run_cumulative_ablation_experiment():
    print("1. Loading data and extracting clean baseline activations...")
    MODEL_PATH = '../models/transformer/tictactoe_model.pth'
    JSON_DATA_PATH = '../data/games.json'

    # Kwargs preserved to avoid TypeError with external script
    activations, relative_train, relative_test, _ = prepare_data(
        sciezka_do_danych='../data/processed/dataset_pelny.pt',
        liczba_trening=1000,
        liczba_test=200
    )

    l1_train_activations = activations['warstwa_1'][:1000].view(-1, 128)

    print("2. Training reference Linear Probe for Spatial Board State (Layer L1)...")
    probe = LinearProbe(128, 27)
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1000):
        optimizer.zero_grad()
        loss = criterion(probe(l1_train_activations).view(-1, 3, 9), relative_train)
        loss.backward()
        optimizer.step()

    l1_test_activations = activations['warstwa_1'][1000:1200].view(-1, 128)
    baseline_acc = evaluate_physics_accuracy(probe, l1_test_activations, relative_test)
    print(f"--> Baseline L1 spatial accuracy: {baseline_acc:.1f}%")

    print("\n3. Formatting test sequences for forward passes through the ablated model...")
    with open(JSON_DATA_PATH, 'r') as f:
        all_games = json.load(f)

    test_games = all_games[1000:1200]
    game_tensors = torch.tensor([get_padded_sequence(game) for game in test_games], dtype=torch.long)

    def extract_ablated_activations(ablated_model):
        """Executes a forward pass and captures ablated L1 activations."""
        extracted_l1 = []

        def hook(module, input_tensor, output_tensor):
            extracted_l1.append(output_tensor.detach())

        handle = ablated_model.transformer.layers[1].register_forward_hook(hook)
        with torch.no_grad():
            ablated_model(game_tensors)
        handle.remove()
        return extracted_l1[0].view(-1, 128)

    print("\n4. Performing single head ablation to establish importance hierarchy...")
    head_ranking = []

    for h in range(8):
        # Initialize a clean model state for each single head ablation
        model = TinyTicTacToeGPT(d_model=128, num_layers=3, nhead=8)
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=True))
        model.eval()

        ablate_attention_heads(model, layer_idx=1, head_list=[h])
        ablated_acts = extract_ablated_activations(model)
        acc = evaluate_physics_accuracy(probe, ablated_acts, relative_test)
        performance_drop = baseline_acc - acc

        head_ranking.append({'head': h, 'acc': acc, 'drop': performance_drop})
        print(f" - L1.H{h} ablated: performance drop of {performance_drop:.2f}% (Acc: {acc:.1f}%)")

    # Sort based on the magnitude of the performance drop (descending)
    head_ranking.sort(key=lambda x: x['drop'], reverse=True)
    sorted_heads = [item['head'] for item in head_ranking]

    print(f"\nAttention head importance hierarchy (most to least critical): {sorted_heads}")

    print("\n5. Executing Cumulative Ablation Sequence...")
    cumulative_results = [baseline_acc]
    ablated_heads_list = []

    for h in sorted_heads:
        ablated_heads_list.append(h)

        model = TinyTicTacToeGPT(d_model=128, num_layers=3, nhead=8)
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=True))
        model.eval()

        ablate_attention_heads(model, layer_idx=1, head_list=ablated_heads_list)
        ablated_acts = extract_ablated_activations(model)
        acc = evaluate_physics_accuracy(probe, ablated_acts, relative_test)
        cumulative_results.append(acc)

        print(f" Ablated {len(ablated_heads_list)} heads {ablated_heads_list} -> Spatial accuracy: {acc:.1f}%")

    print("\n6. Generating ablation curve plot...")
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(9, 6))

    x_axis = np.arange(9)
    ax.plot(x_axis, cumulative_results, marker='o', color='#e74c3c', linewidth=3, markersize=8, zorder=3)

    # Random guessing threshold (~33.3% for 3 classes)
    ax.axhline(33.3, color='gray', linestyle='--', linewidth=2, label='Random Guessing Baseline (~33%)', zorder=1)

    ax.set_title('Cumulative Attention Head Ablation in Layer L1\n(Graceful Degradation Evidence)', fontsize=14,
                 fontweight='bold')
    ax.set_xlabel('Number of Ablated Attention Heads (Ordered by Importance)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Spatial Representation Accuracy (%)', fontsize=12, fontweight='bold')

    ax.set_xticks(x_axis)
    ax.set_ylim(20, 105)
    ax.legend(loc='lower left')

    # Value annotations above data points
    for i, val in enumerate(cumulative_results):
        ax.annotate(f"{val:.1f}%", (x_axis[i], cumulative_results[i] + 2), ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    os.makedirs('../plots', exist_ok=True)
    plot_path = '../plots/08_cumulative_ablation.png'
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved successfully to: {plot_path}")
    plt.show()


if __name__ == "__main__":
    run_cumulative_ablation_experiment()