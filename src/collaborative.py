"""
Neural Collaborative Filtering (NeuMF architecture: He et al. 2017,
adapted for explicit rating regression instead of implicit binary
classification -- we predict the actual star rating, matching the
"predict missing rating in the matrix" approach we've been building
toward).

Architecture: two parallel branches that both take user+item embeddings,
then get combined at the end:
  - GMF branch (Generalized Matrix Factorization): element-wise product
    of user/item embeddings, like classic matrix factorization
  - MLP branch: concatenate user/item embeddings, pass through dense
    layers, lets the model learn non-linear interactions the GMF branch
    can't capture
  - Both branches' outputs get concatenated -> final linear layer ->
    predicted rating

Training strategy for speed on an 8GB GPU:
  - Encode userId/movieId to contiguous 0-indexed integers (needed for
    nn.Embedding lookup tables)
  - Load ALL train/val/test tensors directly onto GPU once (they're only
    a few hundred MB as int32/float32 -- easily fits in 8GB)
  - Batch via GPU-side random permutation indexing instead of a
    DataLoader -- avoids CPU->GPU transfer overhead every batch, which
    is normally the main bottleneck for a model this small
"""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

MODEL_SAVE_PATH = config.MODELS_DIR / "ncf_model.pt"
ID_MAPPING_PATH = config.MODELS_DIR / "ncf_id_mappings.npz"

# ---- hyperparameters ----
EMBED_DIM = 32
MLP_LAYERS = [128, 64, 32]
DROPOUT = 0.35
BATCH_SIZE = 8192
LEARNING_RATE = 1e-3          # bumped slightly since LR scheduler + batchnorm allow it
WEIGHT_DECAY = 1e-5        # small L2 regularization, helps big embedding tables not overfit
NUM_EPOCHS = 20
EARLY_STOP_PATIENCE = 4       # slightly more patient now that LR scheduler can rescue plateaus
TRAIN_FRAC, VAL_FRAC = 0.90, 0.05   # remaining 0.05 is test


class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=EMBED_DIM,
                 mlp_layers=MLP_LAYERS, dropout=DROPOUT,
                 global_mean=3.5):
        super().__init__()

        # separate embedding tables for each branch (standard NeuMF design --
        # sharing them between GMF/MLP branches hurts performance in practice)
        self.gmf_user_embed = nn.Embedding(num_users, embed_dim)
        self.gmf_item_embed = nn.Embedding(num_items, embed_dim)
        self.mlp_user_embed = nn.Embedding(num_users, embed_dim)
        self.mlp_item_embed = nn.Embedding(num_items, embed_dim)

        # Bias terms (classic Netflix Prize SVD++ trick): lets the model
        # explain "this user just rates everything low" or "this movie is
        # universally loved" WITHOUT burning embedding capacity on it, so
        # embeddings can focus purely on taste-matching signal instead.
        # Usually the single biggest RMSE improvement for rating prediction.
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.tensor([global_mean], dtype=torch.float32))

        mlp_blocks = []
        input_dim = embed_dim * 2   # concat user+item embeddings
        for out_dim in mlp_layers:
            mlp_blocks.append(nn.Linear(input_dim, out_dim))
            mlp_blocks.append(nn.BatchNorm1d(out_dim))  # stabilizes training, allows higher LR
            mlp_blocks.append(nn.ReLU())
            mlp_blocks.append(nn.Dropout(dropout))
            input_dim = out_dim
        self.mlp = nn.Sequential(*mlp_blocks)

        # final layer combines GMF output (embed_dim) + MLP output (last layer size)
        self.output_layer = nn.Linear(embed_dim + mlp_layers[-1], 1)

        self._init_weights()

    def _init_weights(self):
        for embed in [self.gmf_user_embed, self.gmf_item_embed,
                      self.mlp_user_embed, self.mlp_item_embed]:
            nn.init.normal_(embed.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, user_idx, item_idx):
        gmf_out = self.gmf_user_embed(user_idx) * self.gmf_item_embed(item_idx)

        mlp_in = torch.cat([
            self.mlp_user_embed(user_idx),
            self.mlp_item_embed(item_idx),
        ], dim=-1)
        mlp_out = self.mlp(mlp_in)

        combined = torch.cat([gmf_out, mlp_out], dim=-1)
        interaction = self.output_layer(combined).squeeze(-1)

        bias = (
            self.global_bias
            + self.user_bias(user_idx).squeeze(-1)
            + self.item_bias(item_idx).squeeze(-1)
        )
        return interaction + bias


def prepare_data(device):
    """
    Load cleaned ratings, encode ids to contiguous integers, split
    90/5/5, move everything to GPU tensors.
    """
    ratings = pd.read_parquet(config.DATA_PROCESSED / "ratings_clean.parquet")

    unique_users = ratings["userId"].unique()
    unique_movies = ratings["movieId"].unique()
    user_to_idx = {u: i for i, u in enumerate(unique_users)}
    movie_to_idx = {m: i for i, m in enumerate(unique_movies)}

    np.savez(
        ID_MAPPING_PATH,
        user_ids=unique_users,
        movie_ids=unique_movies,
    )

    ratings["user_idx"] = ratings["userId"].map(user_to_idx).astype(np.int64)
    ratings["movie_idx"] = ratings["movieId"].map(movie_to_idx).astype(np.int64)

    # shuffle once, then split 90/5/5
    ratings = ratings.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n = len(ratings)
    train_end = int(n * TRAIN_FRAC)
    val_end = train_end + int(n * VAL_FRAC)

    def to_tensors(df_slice):
        u = torch.tensor(df_slice["user_idx"].to_numpy(), dtype=torch.long, device=device)
        m = torch.tensor(df_slice["movie_idx"].to_numpy(), dtype=torch.long, device=device)
        r = torch.tensor(df_slice["rating"].to_numpy(), dtype=torch.float32, device=device)
        return u, m, r

    train = to_tensors(ratings.iloc[:train_end])
    val = to_tensors(ratings.iloc[train_end:val_end])
    test = to_tensors(ratings.iloc[val_end:])

    global_mean = float(ratings.iloc[:train_end]["rating"].mean())

    print(f"Train: {len(train[0]):,} | Val: {len(val[0]):,} | Test: {len(test[0]):,}")
    print(f"Users: {len(unique_users):,} | Movies: {len(unique_movies):,}")
    print(f"Global mean rating (train): {global_mean:.3f}")

    return train, val, test, len(unique_users), len(unique_movies), global_mean


def run_epoch(model, data, optimizer=None, scaler=None, batch_size=BATCH_SIZE):
    """
    If optimizer is provided, does a training pass (with backprop + mixed
    precision via GradScaler). If optimizer is None, does an evaluation
    pass (no grad, no shuffle needed).
    Returns RMSE and MAE for the epoch.
    """
    users, movies, ratings = data
    n = len(users)
    is_train = optimizer is not None
    device_type = users.device.type

    model.train() if is_train else model.eval()
    loss_fn = nn.MSELoss()

    perm = torch.randperm(n, device=users.device) if is_train else torch.arange(n, device=users.device)

    total_sq_err, total_abs_err, total_count = 0.0, 0.0, 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]

            # BatchNorm breaks on batch size 1 during training -- skip the
            # (rare) leftover batch of size 1 rather than crash
            if is_train and len(idx) < 2:
                continue

            u_batch, m_batch, r_batch = users[idx], movies[idx], ratings[idx]

            with torch.autocast(device_type=device_type, enabled=(device_type == "cuda")):
                preds = model(u_batch, m_batch)
                loss = loss_fn(preds, r_batch)

            if is_train:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

            batch_n = len(idx)
            total_sq_err += ((preds.float() - r_batch) ** 2).sum().item()
            total_abs_err += (preds.float() - r_batch).abs().sum().item()
            total_count += batch_n

    rmse = (total_sq_err / total_count) ** 0.5
    mae = total_abs_err / total_count
    return rmse, mae


def train_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")
    if device == "cpu":
        print("WARNING: no GPU detected, this will be much slower than the time estimates.")

    train, val, test, num_users, num_movies, global_mean = prepare_data(device)

    model = NeuMF(num_users, num_movies, global_mean=global_mean).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best_val_rmse = float("inf")
    epochs_without_improvement = 0

    total_start = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()

        train_rmse, train_mae = run_epoch(model, train, optimizer, scaler)
        val_rmse, val_mae = run_epoch(model, val, optimizer=None)
        scheduler.step(val_rmse)

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:2d}/{NUM_EPOCHS} | "
              f"train RMSE {train_rmse:.4f} MAE {train_mae:.4f} | "
              f"val RMSE {val_rmse:.4f} MAE {val_mae:.4f} | "
              f"lr {current_lr:.2e} | {epoch_time:.1f}s")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_users": num_users,
                "num_movies": num_movies,
                "embed_dim": EMBED_DIM,
                "mlp_layers": MLP_LAYERS,
                "global_mean": global_mean,
            }, MODEL_SAVE_PATH)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"Early stopping -- no val improvement for {EARLY_STOP_PATIENCE} epochs")
                break

    total_time = time.time() - total_start
    print(f"\nTotal training time: {total_time / 60:.1f} minutes")

    # final test evaluation using the BEST saved checkpoint, not the last epoch
    checkpoint = torch.load(MODEL_SAVE_PATH, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_rmse, test_mae = run_epoch(model, test, optimizer=None)
    print(f"Final TEST results -- RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")
    print(f"Model saved to {MODEL_SAVE_PATH}")

    return model


if __name__ == "__main__":
    train_model()