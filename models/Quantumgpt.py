import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from ..core.quantum_field import QuantumField

class QuantumGPT(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 512, d_model))
        decoder_layer = TransformerDecoderLayer(d_model, nhead)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.quantum_field = QuantumField(field_dims=[1])

    def forward(self, tgt, memory=None):
        embed = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]
        qfield_out = self.quantum_field(embed)
        output = self.transformer_decoder(qfield_out, memory) if memory is not None else self.transformer_decoder(qfield_out, qfield_out)
        logits = self.fc_out(output)
        return logits
