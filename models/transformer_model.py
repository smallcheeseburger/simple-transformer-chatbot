import torch.nn as nn
class transformer(nn.Module):
    def __init__(self,vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer = nn.Transformer(
            d_model = embedding_dim,
            nhead = 4,
            num_encoder_layers = 6,
            num_decoder_layers = 6,
            dim_feedforward = 4 * embedding_dim,
            dropout = 0.3,
            batch_first = True
        )
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

    def forward(self,encoder, decoder):
        encoder_embedding = self.embedding(encoder)
        decoder_embedding = self.embedding(decoder)
        seq_len = decoder.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(decoder.device)

        transformer_out = self.transformer(
            src=encoder_embedding,
            tgt=decoder_embedding,
            tgt_mask=tgt_mask
        )

        out = self.fc_out(transformer_out)
        return out
