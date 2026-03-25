import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer,IEncoderLayer,IEncoder
from layers.SelfAttention_Family import M_FullAttention, AttentionLayer
from layers.Embed import DataEmbedding

class EncoderLayer(torch.nn.Module):
    def __init__(self, num_hidden, ffn_hidden, seq_length, heads=1, dropout=0.5):
        super(EncoderLayer, self).__init__()
        self.query = nn.Linear(num_hidden, num_hidden)
        self.key = nn.Linear(num_hidden, num_hidden)
        self.value = nn.Linear(num_hidden, num_hidden)
        self.attn = nn.MultiheadAttention(embed_dim=num_hidden, num_heads=heads, dropout=dropout)
        self.attn2 = nn.MultiheadAttention(embed_dim=seq_length, num_heads=5,
                                           dropout=dropout)  # num_hidden必须是heads的倍数
        self.norm1 = nn.LayerNorm(num_hidden)
        self.fc1 = nn.Linear(num_hidden, ffn_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(ffn_hidden, num_hidden)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(num_hidden)

    def forward(self, X):
        # Q = self.query(X)
        # K = self.key(X)
        # V = self.value(X)
        # Y, _ = self.attn(Q,K,V)
        X = X.permute(1, 0, 2)
        Y, attn1 = self.attn(X, X, X)
        X = X.permute(1, 0, 2)
        Y = Y.permute(1, 0, 2)
        # x_texts = [str(x) for x in range(1, 46)]
        # for i in range(100):
        #     attention_plot(attn1[i].cpu().detach().numpy(), x_texts, x_texts, figsize=(45, 45), figure_path='./figures',
        #                figure_name='Unit {} seq_attention_weight.png'.format(i+1))
        Y = Y.permute(2, 0, 1)
        Y, attn2 = self.attn2(Y, Y, Y)
        Y = Y.permute(1, 2, 0)
        # x_texts = [str(x) for x in range(1, 17)]
        # for i in range(100):
        #     attention_plot(attn2[i].cpu().detach().numpy(), x_texts, x_texts, annot=False, figsize=(16, 16), figure_path='./figures_fea',
        #                figure_name='Unit {} fea_attention_weight.png'.format(i+1))

        X = self.norm1(X + self.dropout(Y))
        Y = self.fc2(self.relu(self.fc1(X)))
        X = self.norm2(X + self.dropout(Y))
        return X


class MLP(nn.Module):
    def __init__(self, num_inputs, num_hiddens):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, 1))

    def forward(self, X):
        return self.mlp(X)


class TwoP_Transformer(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs,input_feature):
        super(TwoP_Transformer, self).__init__()
        self.seq_len = configs.input_length
        self.dropout = 0.1
        self.channels = input_feature

        self.linear = nn.Linear(self.channels, configs.d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_len, configs.d_model))
        # Initiate Time_step encoder
        self.encoder = nn.Sequential()
        for i in range(configs.d_layers):
            self.encoder.add_module(f"{i}", EncoderLayer(configs.d_model, configs.d_model, self.seq_len, configs.n_heads, self.dropout))

        self.out = MLP(configs.d_model, configs.d_model)

    def forward(self, x_enc,**kwargs):

        X = self.linear(x_enc) + self.pos_embedding  # ((batch_size,seq_len,num_hidden))

        # time step encoder
        for enc_layer in self.encoder:
            X = enc_layer(X)

        Y = self.out(X[:, -1, :])

        return None,Y #[B,1]


















