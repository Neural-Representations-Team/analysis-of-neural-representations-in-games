# analysis-of-neural-representations-in-games
The project aims to investigate how models based on the Transformer architecture develop an internal understanding of the rules and world model by learning solely from flat sequences of data.

## Project Structure

```text
.
├── data/                   # Dataset storage (Ignored by git)
│   ├── raw/                # Original game transcripts (synthetic/championship) 
│   └── processed/          # Extracted activations and board states [cite: 151]
├── models/                 # Saved model weights
│   ├── transformer/        # Trained GPT-variant weights [cite: 114]
│   └── probes/             # Trained linear and non-linear probes [cite: 179]
├── scripts/                # Source code for the project
│   ├── data_gen.py         # Script to generate Tic-Tac-Toe games
│   ├── train_model.py      # Main transformer training script (Dominik)
│   ├── hooks.py            # Activation extraction using forward hooks (Szymon) [cite: 191]
│   ├── train_probes.py     # Linear probe training logic (Szymon) [cite: 148]
│   └── eval_interventions.py # Causal intervention experiments [cite: 300]
├── notebooks/              # Interactive analysis and visualization
│   ├── activation_viz.ipynb # Exploratory data analysis of internal states
│   └── saliency_maps.ipynb # Generation of latent saliency maps (Wiktor) [cite: 315]
├── tests/                  # Unit tests for hooks and model logic
├── .gitignore              # Files to be excluded from version control (data/, models/)
├── requirements.txt        # Project dependencies (torch, wandb, etc.)
└── README.md               # Project overview and setup instructions