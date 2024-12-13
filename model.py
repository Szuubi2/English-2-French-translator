import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader



def getTransformerModel(DEVICE, EMBEDDING_DIM, MAX_CONTEXT_LEN, NUM_OF_HEADS, 
                        DROPOUT, NUM_OF_BLOCKS, HEAD_SIZE, VOCAB_SIZE_TRG, 
                        VOCAB_SIZE_SRC, PAD_TOKEN):

    class AttentionHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.key_emb = nn.Linear(EMBEDDING_DIM, HEAD_SIZE)
            self.query_emb = nn.Linear(EMBEDDING_DIM, HEAD_SIZE)
            self.value_emb = nn.Linear(EMBEDDING_DIM, HEAD_SIZE)
            self.dropout = nn.Dropout(DROPOUT)

        def forward(self, k, q, v, mask):
            # A = Softmax((QK.T)/root(dk))V
            keys = self.key_emb(k)
            queries = self.query_emb(q)
            values = self.value_emb(v)

            score_matrix = queries @ keys.transpose(-1, -2) # (B, T_decoder, T_encoder)
            score_matrix = score_matrix / (HEAD_SIZE**0.5)

            mask = mask.view(mask.size(0), -1, mask.size(1))
            score_matrix = torch.where(mask == 0, -torch.inf, score_matrix)
            score_matrix = torch.softmax(score_matrix, -1)

            return self.dropout(score_matrix @ values) # (B, T_decoder, T_encoder) @ (B, T_encoder, HEAD_SIZE) => (B, T, HEAD_SIZE)


    class MultiHeadAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention_heads = nn.ModuleList([AttentionHead() for _ in range(NUM_OF_HEADS)])
            self.projection = nn.Linear(NUM_OF_HEADS * HEAD_SIZE, EMBEDDING_DIM)
            self.dropout = nn.Dropout(DROPOUT)

        def forward(self, k, q, v, mask):
            heads_out = torch.cat([head(k, q, v, mask) for head in self.attention_heads], dim = -1)
            heads_out = self.projection(heads_out)
            return self.dropout(heads_out) #(B, T, EMBEDDING_DIM)


    class FeedForward(nn.Module):
        def __init__(self):
            super().__init__()
            self.fwd = nn.Sequential(nn.Linear(EMBEDDING_DIM, 3 * EMBEDDING_DIM),
                                    nn.ReLU(),
                                    nn.Linear(3 * EMBEDDING_DIM, EMBEDDING_DIM), 
                                    nn.Dropout(DROPOUT))

        def forward(self, x):
            return self.fwd(x) # (B, T, EMBEDDING_DIM)


    class EncoderBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_norm_1 = nn.LayerNorm(EMBEDDING_DIM)
            self.self_attention = MultiHeadAttention()
            self.layer_norm_2 = nn.LayerNorm(EMBEDDING_DIM)
            self.feed_forward = FeedForward()

        def forward(self, x, mask):
            x = self.layer_norm_1(x + self.self_attention(x, x, x, mask))
            x = self.layer_norm_2(x + self.feed_forward(x))
            return x # (B, T, EMBEDDING_DIM)



    class TransformerEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(VOCAB_SIZE_SRC, EMBEDDING_DIM)
            self.pos_emb = nn.Embedding(MAX_CONTEXT_LEN, EMBEDDING_DIM)
            self.blocks = nn.ModuleList([EncoderBlock() for _ in range(NUM_OF_BLOCKS)])

        def forward(self, x, mask):
            B, T = x.shape
            x = self.tok_emb(x) + self.pos_emb(torch.arange(T, device=DEVICE))
            
            for block in self.blocks:
                x = block(x, mask)

            return x # (B, T, EMBEDDING_DIM)


    class DecoderBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_norm_1 = nn.LayerNorm(EMBEDDING_DIM)
            self.attention_masked = MultiHeadAttention()
            self.layer_norm_2 = nn.LayerNorm(EMBEDDING_DIM)
            self.attention = MultiHeadAttention()
            self.layer_norm_3 = nn.LayerNorm(EMBEDDING_DIM)
            self.feed_forward = FeedForward() 

        def forward(self, x, enc_out, mask_src, mask_trg):
            x = self.layer_norm_1(x + self.attention_masked(x, x, x, mask_trg))
            x = self.layer_norm_2(x + self.attention(enc_out, x, enc_out, mask_src))
            x = self.layer_norm_3(x + self.feed_forward(x))
            return x # (B, T, EMBEDDING_DIM)
        


    class TransformerDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(num_embeddings = VOCAB_SIZE_TRG, embedding_dim = EMBEDDING_DIM)
            self.pos_emb = nn.Embedding(MAX_CONTEXT_LEN, EMBEDDING_DIM)
            self.blocks = nn.ModuleList([DecoderBlock() for _ in range(NUM_OF_BLOCKS)])
            self.head = nn.Linear(EMBEDDING_DIM, VOCAB_SIZE_TRG)

        def forward(self, x, enc_out, mask_src, mask_trg):
            B, T = x.shape
            x = self.tok_emb(x) + self.pos_emb(torch.arange(T, device=DEVICE))

            for block in self.blocks:
                x = block(x, enc_out, mask_src, mask_trg)

            x = self.head(x) 
            return x # (B, T, VOCAB_SIZE_DECODER)



    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = TransformerEncoder()
            self.decoder = TransformerDecoder()
            self.apply(self._init_weights)

        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        def create_trg_mask(self, trg, trg_pad_mask):
            T = trg.shape[1]
            trg_mask = trg_pad_mask
            nopeak_mask = torch.tril(torch.ones((T, T), device=DEVICE)).unsqueeze(0).bool()
            trg_mask = trg_mask.unsqueeze(-2) & nopeak_mask
            return trg_mask  # (B, T, T)

        def forward(self, x, y, src_pad_mask, trg_pad_mask):
            # src: (B, T_src)
            # trg: (B, T_trg)
            
            src_mask = src_pad_mask.to(DEVICE)
            trg_mask = self.create_trg_mask(y, trg_pad_mask).to(DEVICE)

            enc_out = self.encoder(x, src_mask)
            out = self.decoder(y, enc_out, src_mask, trg_mask)
            return out



    return Transformer()


