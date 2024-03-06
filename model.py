import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import GPT2Model, GPT2Config
import math

class GPT(nn.Module):
    def __init__(self, n_dims, n_positions=40, n_embd=64, n_layer=2, n_head=1, n_out=2, dropout_rate=0):
        super(GPT, self).__init__()
        configuration = GPT2Config(
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
            output_attentions=True,
        )

        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, n_out)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        embeds = self._read_in(x)
        model_outputs = self._backbone(inputs_embeds=embeds)
        decoder_output = model_outputs.last_hidden_state
        decoder_output = self.dropout(decoder_output) 
        attentions = model_outputs.attentions 
        output = self._read_out(decoder_output)
        return output, attentions