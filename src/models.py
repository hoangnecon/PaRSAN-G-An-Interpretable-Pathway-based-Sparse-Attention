import torch
import torch.nn as nn
from sparsemax import Sparsemax
import numpy as np
import math

# --- Helper function để tạo MLP linh hoạt ---
def _create_mlp(input_dim, output_dim, hidden_layers, dropout_rate):
    """Tạo ra một mạng Multi-Layer Perceptron (MLP) với kiến trúc tùy chỉnh."""
    layers = []
    current_dim = input_dim
    for hidden_dim in hidden_layers:
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)

# --- v1: Mô hình PaRSAN gốc ---
class PaRSAN(nn.Module):
    def __init__(self, input_dim, attention_dim, dropout_rate, num_classes):
        super(PaRSAN, self).__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, input_dim)
        )
        self.sparsemax = Sparsemax(dim=1)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        attention_weights = self.sparsemax(self.attention_net(x))
        weighted_features = x * attention_weights
        output = self.classifier(weighted_features)
        return output

class PaRSAN_G(nn.Module):
    def __init__(self, input_dim, num_classes,
                 attention_hidden_layers, gate_hidden_layers, classifier_hidden_layers,
                 dropout_rate):
        super(PaRSAN_G, self).__init__()
        self.attention_net = _create_mlp(input_dim, input_dim, attention_hidden_layers, dropout_rate)
        self.sparsemax = Sparsemax(dim=1)
        self.gate_net = _create_mlp(input_dim, input_dim, gate_hidden_layers, dropout_rate)
        self.classifier = _create_mlp(input_dim, num_classes, classifier_hidden_layers, dropout_rate)

    def forward(self, x, return_attention=False):
        attention_scores = self.attention_net(x)
        attention_weights = self.sparsemax(attention_scores)
        attended_features = x * attention_weights

        gate_scores = self.gate_net(x)
        gate_values = torch.sigmoid(gate_scores)

        gated_features = (gate_values * attended_features) + ((1 - gate_values) * x)
        output = self.classifier(gated_features)

        if return_attention:
            return output, attention_weights

        return output
