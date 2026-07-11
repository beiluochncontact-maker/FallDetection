import copy
import math

import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Dataset
class FallDataset(Dataset):

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        data = torch.tensor(
            sample["window"],
            dtype=torch.float32
        )
        label = torch.tensor(
            sample["label"],
            dtype=torch.long
        )
        return data, label

# Positional Encoding
class PositionalEncoding(nn.Module):

    def __init__(self,
                 d_model,
                 dropout=0.1,
                 max_len=5000):

        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(
            0,
            max_len,
            dtype=torch.float32
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2
            ).float()
            *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# Transformer Model
class FallTransformer(nn.Module):

    def __init__(self,
                 input_dim,
                 d_model,
                 nhead,
                 num_layers,
                 dim_feedforward,
                 dropout):

        super().__init__()

        self.embedding = nn.Linear(
            input_dim,
            d_model
        )

        self.position = PositionalEncoding(
            d_model,
            dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Sequential(

            nn.Linear(d_model,64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.position(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)


def train_transformer(
        train_set,
        val_set,
        param_grid):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    train_loader = DataLoader(
        FallDataset(train_set),
        batch_size=param_grid["batch_size"],
        shuffle=True
    )

    val_loader = DataLoader(
        FallDataset(val_set),
        batch_size=param_grid["batch_size"],
        shuffle=False
    )

    model = FallTransformer(
        input_dim=param_grid["input_dim"],
        d_model=param_grid["d_model"],
        nhead=param_grid["nhead"],
        num_layers=param_grid["num_layers"],
        dim_feedforward=param_grid["dim_feedforward"],
        dropout=param_grid["dropout"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param_grid["learning_rate"]
    )

    best_model = None
    best_acc = -1

    for epoch in range(param_grid["epochs"]):

        model.train()

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        model.eval()
        pred_list = []
        label_list = []

        with torch.no_grad():
            for x, y in val_loader:

                x = x.to(device)
                output = model(x)
                pred = torch.argmax(
                    output,
                    dim=1
                )

                pred_list.extend(
                    pred.cpu().numpy()
                )

                label_list.extend(
                    y.numpy()
                )

        val_acc = accuracy_score(
            label_list,
            pred_list
        )

        print(
            f"Epoch {epoch+1:02d}"
            f"  Validation={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_model = copy.deepcopy(model)

    return {
        "best_model": best_model, 
        "best_acc": best_acc
    }


def evaluate_transformer(
        model,
        test_set):
    
    device = next(
        model.parameters()
    ).device

    test_loader = DataLoader(
        FallDataset(test_set),
        batch_size=64,
        shuffle=False
    )

    model.eval()

    pred_list = []
    label_list = []

    with torch.no_grad():
        for x, y in test_loader:

            x = x.to(device)
            output = model(x)
            pred = torch.argmax(
                output,
                dim=1
            )

            pred_list.extend(
                pred.cpu().numpy()
            )

            label_list.extend(
                y.numpy()
            )

    result = {

        "accuracy": accuracy_score(
            label_list,
            pred_list
        ),

        "precision": precision_score(
            label_list,
            pred_list,
            zero_division=0
        ),

        "recall": recall_score(
            label_list,
            pred_list,
            zero_division=0
        ),

        "f1": f1_score(
            label_list,
            pred_list,
            zero_division=0
        ),

        "confusion_matrix": confusion_matrix(
            label_list,
            pred_list
        )

    }

    return result