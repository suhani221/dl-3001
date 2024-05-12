# baseline transformer for NeuroGNN that didn't work out because of dimension error and time constraint. Class added to the baseline file of NeuroGNN and called in train.py.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
import math

class MultiHeadAttention(Module):
    def __init__(self, d_model, q, v, h, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.q = q
        self.v = v
        self.h = h
        self.d_k = d_model // h
        
        self.W_q = nn.Linear(d_model, q * h)
        self.W_k = nn.Linear(d_model, q * h)
        self.W_v = nn.Linear(d_model, v * h)
        self.W_o = nn.Linear(v * h, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Linear projections
        Q = self.W_q(x).view(batch_size, -1, self.h, self.q).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.h, self.q).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.h, self.v).transpose(1, 2)
        
        # Attention function
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, V).transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.v)
        
        return self.W_o(output)

class FeedForward(Module):
    def __init__(self, d_model, d_hidden, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_hidden, d_model)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        return self.linear2(x)

class Encoder(Module):
    def __init__(self, d_model, d_hidden, q, v, h, dropout=0.1):
        super(Encoder, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, q, v, h, dropout)
        self.ff = FeedForward(d_model, d_hidden, dropout)
        
    def forward(self, x):
        x = self.norm1(x + self.mha(x))
        x = self.norm2(x + self.ff(x))
        return x

class Transformer(Module):
    def __init__(self, d_model, d_hidden, q, v, h, num_classes, num_layers, dropout, d_channel, d_hz, d_output, d_input):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([Encoder(d_model, d_hidden, q, v, h, dropout) for _ in range(num_layers)])
        self.embedding_step = nn.Linear(d_channel, d_model)
        self.embedding_channel = nn.Linear(d_input * d_hz, d_model)
        self.embedding_hz = nn.Linear(d_input * d_channel, d_model)
        self.gate = nn.Linear(d_model * 3, 3)
        self.output_linear = nn.Linear(d_model * 3, d_output)
        
    def forward(self, x):
        # Assuming x dimensions are (batch_size, seq_length, num_channels, feature_size)
        batch_size, seq_length, num_channels, feature_size = x.shape
        x = x.view(batch_size * seq_length, num_channels, feature_size)  # Flatten batch and seq_length
        
        # Embeddings
        step_x = self.embedding_step(x)
        channel_x = self.embedding_channel(x.view(batch_size * seq_length, -1))
        hz_x = self.embedding_hz(x.view(batch_size * seq_length, -1))
        
        # Process through encoders
        for layer in self.layers:
            step_x = layer(step_x)
            channel_x = layer(channel_x)
            hz_x = layer(hz_x)
        
        # Concatenate and apply gate
        concatenated = torch.cat([step_x, channel_x, hz_x], dim=-1)
        gate = F.softmax(self.gate(concatenated), dim=-1)
        gated_output = concatenated * gate.unsqueeze(-1).expand(-1, -1, concatenated.size(-1))

        # Final output
        output = self.output_linear(gated_output.view(batch_size, seq_length, -1))

        return output
     
     
     
# add to train.py of the model
Transformer(d_model=512, d_hidden=1024, q=64, v=64, h=8, num_classes=4, num_layers=3, dropout=0.5, d_channel=25, d_hz=244, d_output=4, d_input=32)
# batch = 40
# seq = 60
