import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


# --- MODEL DEFINITION ---
class TinyTicTacToeGPT(nn.Module):
    def __init__(self, d_model=128, num_layers=3, nhead=8):
        super().__init__()
        self.embedding = nn.Embedding(11, d_model)
        self.pos_encoder = nn.Embedding(10, d_model)
        # NOTE: Using batch_first=True to facilitate data manipulation
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


def generate_attention_heatmaps(model_path, game_sequence, target_head=5, target_layer=1):
    """
    Extracts and visualizes attention weights from a specified head.
    Defaults to L1.H5, which evidence supports as structurally significant.
    """
    print(f"Initializing analysis for game sequence: {game_sequence}")

    # 1. Load the trained model
    device = torch.device('cpu')
    model = TinyTicTacToeGPT(d_model=128, num_layers=3, nhead=8).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 2. Prepare the input tensor (1 x Sequence Length)
    input_tensor = torch.tensor([game_sequence], dtype=torch.long).to(device)
    seq_len = input_tensor.size(1)

    # Implement mask to maintain consistency with the model's inference process
    mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)

    print("Preparing a dedicated forward pass to extract raw attention weights...")

    # 3. Access the specified layer to isolate attention matrices
    target_module = model.transformer.layers[target_layer]

    with torch.no_grad():
        positions = torch.arange(0, seq_len, device=device).unsqueeze(0)
        embedded_input = model.embedding(input_tensor) + model.pos_encoder(positions)

        # Process data through the preceding layers
        temp_out = embedded_input
        for i in range(target_layer):
            temp_out = model.transformer.layers[i](temp_out, src_mask=mask)

        # Isolate MultiheadAttention outputs
        # Setting average_attn_weights=False returns weights for all attention heads
        attn_out, attn_weights_raw = target_module.self_attn(
            temp_out, temp_out, temp_out, attn_mask=mask, average_attn_weights=False
        )

    # Tensor dimensions: [Batch, Head_Count, Target_Sequence, Source_Sequence]
    target_weights = attn_weights_raw[0, target_head].numpy()

    # 4. Generate the Heatmap Plot
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")

    # Axis labels indicating the game moves
    move_labels = [f"Move {i + 1}\n(Square {game_sequence[i]})" for i in range(seq_len)]

    sns.heatmap(
        target_weights,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        cbar_kws={'label': 'Attention Weight'},
        xticklabels=move_labels,
        yticklabels=move_labels
    )

    plt.title(
        f"Attention Pattern - Layer L{target_layer}, Head H{target_head}\nGame Sequence: {game_sequence}",
        fontsize=14, fontweight="bold"
    )
    plt.xlabel("Source Token (Attending TO)", fontsize=12)
    plt.ylabel("Destination Token (Attending FROM)", fontsize=12)

    plt.tight_layout()

    # Save the output
    os.makedirs('../plots/heatmaps', exist_ok=True)
    filename = f"../plots/heatmaps/L{target_layer}_H{target_head}_{''.join(map(str, game_sequence))}.png"
    plt.savefig(filename, dpi=300)
    print(f"Plot saved to: {filename}")
    plt.show()


if __name__ == "__main__":
    MODEL_PATH = '../models/transformer/tictactoe_model.pth'

    # Analyze the winning game sequence
    test_game = [0, 1, 4, 8, 3, 5, 6]

    generate_attention_heatmaps(MODEL_PATH, test_game, target_head=5, target_layer=1)