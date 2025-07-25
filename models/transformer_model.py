import torch.nn as nn
import torch
import math

class PositionalEndocing(nn.Module):
    def __init__(self, model_dim, max_len = 100):
        super().__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len,dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        pe = pe.unsqueeze(0) # [1, max_len, model_dim]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:,:x.size(1)]
        return x

class transformer(nn.Module):
    def __init__(self,vocab_size, embedding_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEndocing(embedding_dim)
        self.pos_decoder = PositionalEndocing(embedding_dim)
        self.pad_idx = pad_idx
        self.encoder_dropout = nn.Dropout(0.1)
        self.decoder_dropout = nn.Dropout(0.3)
        self.transformer = nn.Transformer(
            d_model = embedding_dim,
            nhead = 8,
            num_encoder_layers = 8,
            num_decoder_layers = 8,
            dim_feedforward = 4 * embedding_dim,
            dropout = 0.3,
            batch_first = True
        )
        self.fc_out = nn.Linear(embedding_dim, vocab_size)
        self.norm_encoder = nn.LayerNorm(embedding_dim)
        self.norm_decoder = nn.LayerNorm(embedding_dim)
        self.norm_output = nn.LayerNorm(embedding_dim)

    def forward(self,encoder, decoder):
        encoder_embedding = self.embedding(encoder)
        encoder_embedding = self.pos_encoder(encoder_embedding)
        encoder_embedding = self.norm_encoder(encoder_embedding) 
        encoder_embedding = self.encoder_dropout(encoder_embedding)

        decoder_embedding = self.embedding(decoder)
        decoder_embedding = self.pos_decoder(decoder_embedding)
        decoder_embedding = self.norm_decoder(decoder_embedding)
        decoder_embedding = self.decoder_dropout(decoder_embedding)

        seq_len = decoder.size(1)
        tgt_mask = self.transformer.generate_square_subsequent_mask(seq_len).to(decoder.device).float()

        src_key_padding_mask = (encoder == self.pad_idx).float()
        tgt_key_padding_mask = (decoder == self.pad_idx).float()

        transformer_out = self.transformer(
            src=encoder_embedding,
            tgt=decoder_embedding,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )

        transformer_out = self.norm_output(transformer_out)

        out = self.fc_out(transformer_out)
        return out
