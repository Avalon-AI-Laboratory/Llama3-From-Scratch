import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple

from .roformer_helper import precompute_freqs_cis, apply_rotary_emb, repeat_kv

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_mult: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_sz: int = 32
    max_seq_len: int = 2048
    device: str = "cuda"

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.scale
    
# class Attention(nn.Module):
#     def __init__(self, args: ModelArgs):
#         super().__init__()
#         self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
#         self.n_local_heads = args.n_heads
#         self.n_local_kv_heads = self.n_kv_heads
#         self.n_rep = self.n_local_heads // self.n_local_kv_heads
#         self.head_dim = args.dim // args.n_heads

#         self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
#         self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
#         self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
#         self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

#         self.cache_k = torch.zeros(
#             (
#                 args.max_batch_sz,
#                 args.max_seq_len,
#                 self.n_local_kv_heads,
#                 self.head_dim,
#             )
#         ).cuda()
#         self.cache_v = torch.zeros(
#             (
#                 args.max_batch_sz,
#                 args.max_seq_len,
#                 self.n_local_kv_heads,
#                 self.head_dim,
#             )
#         ).cuda()

#     def forward(
#         self,
#         x: torch.Tensor,
#         start_pos: int,
#         freqs_cis: torch.Tensor,
#         mask: Optional[torch.Tensor],
#         inference: bool = False
#     ):
#         # ##### Experimental #####
#         if not inference:
#             self.cache_k.detach()
#             self.cache_v.detach()
#         # ########################
#         bsz, seqlen, _ = x.shape

#         # Compute query, key, and value projections
#         xq = self.wq(x)
#         xk = self.wk(x)
#         xv = self.wv(x)

#         xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
#         xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
#         xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

#         # Apply rotary position embeddings
#         xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

#         # Move cache to the same device as input
#         self.cache_k = self.cache_k.to(xq)
#         self.cache_v = self.cache_v.to(xq)

#         # Update cache with new key and value projections
#         self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
#         self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

#         # Get cached keys and values
#         keys = self.cache_k[:bsz, : start_pos + seqlen]
#         values = self.cache_v[:bsz, : start_pos + seqlen]

#         # Repeat keys/values if needed
#         keys = repeat_kv(keys, self.n_rep)
#         values = repeat_kv(values, self.n_rep)

#         # Transpose for attention computation
#         xq = xq.transpose(1, 2)
#         keys = keys.transpose(1, 2)
#         values = values.transpose(1, 2)

#         # Compute attention scores
#         scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
#         if mask is not None:
#             scores = scores + mask
#         scores = F.softmax(scores.float(), dim=-1).type_as(xq)

#         # Compute attention output
#         output = torch.matmul(scores, values)
#         output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        
#         # Final projection
#         return self.wo(output)

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        # Embedding dimension
        self.dim = args.dim
        # Number of heads assigned to Query
        self.n_heads = args.n_heads
        # Number of heads assigned to Key and values. If "None", the number will be same as Query.
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # Dimension of each head relative to model dimension
        self.head_dim = args.dim // args.n_heads
        # Number of repetition in order to make time Key, Value heads to match Query heads number
        self.n_rep = args.n_heads // args.n_kv_heads

        # Weight initialize for Keys, Querys, Values and Oupt. Notice that the out_feature value of weight for q and kv are based on it's heads
        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False, device=args.device)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False, device=args.device)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False, device=args.device)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False, device=args.device)

        # Initialize caches to store Key, Values at start. (KV Cache Implementation)
        self.cache_k = torch.zeros((args.max_batch_sz, args.max_seq_len, self.n_kv_heads, self.head_dim), device=args.device)
        self.cache_v = torch.zeros((args.max_batch_sz, args.max_seq_len, self.n_kv_heads, self.head_dim), device=args.device)

    def forward(self, x: torch.Tensor, start_pos, inference):
        # Shape of the input embedding: [bsz,seq_len,dim]
        bsz, seq_len, _ = x.shape
        # Mask will be used during 'Training' and is not required for 'inference' due to the use of KV cache.
        mask = None

        xq = self.wq(x)  #x[bsz,seq_len,dim]*wq[dim,n_heads * head_dim] -> q[bsz,seq_len,n_heads * head_dim]
        xk = self.wk(x)  #x[bsz,seq_len,dim]*wq[dim,n_kv_heads * head_dim] -> k[bsz,seq_len,n_kv_heads * head_dim]
        xv = self.wv(x)  #x[bsz,seq_len,dim]*wq[dim,n_kv_heads * head_dim] -> v[bsz,seq_len,n_kv_heads * head_dim]

        # Reshaping Querys, Keys and Values by their number of heads. (Group Query Attention Implementation)
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)      #xq[bsz,seq_len,n_heads, head_dim]
        xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)   #xk[bsz,seq_len,n_kv_heads, head_dim]
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)   #xv[bsz,seq_len,n_kv_heads, head_dim]

        # Model - Inference Mode: kv-cache is enabled at inference mode only.
        if inference:
            # Compute rotation matrix for each position in the sequence
            freqs_cis = precompute_freqs_cis(dim=self.head_dim, end=self.args.max_seq_len * 2)
            # During inferencing, we should only take the rotation matrix range from the current position of the tokens.
            freqs_cis = freqs_cis[start_pos : start_pos + seq_len]
            # Apply RoPE to Queries and Keys embeddings
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)
            # Store Keys and Values token embedding into their respective cache [KV Cache Implementation]
            self.cache_k[:bsz, start_pos:start_pos + seq_len] = xk
            self.cache_v[:bsz, start_pos:start_pos + seq_len] = xv

            # Assign all the previous tokens embeddings upto current tokens position to Keys and Values variable for Attention Calculation
            keys = self.cache_k[:bsz, :start_pos + seq_len]
            values = self.cache_v[:bsz, :start_pos + seq_len]

            # At this point, they Keys and Values shape aren't same with Queries Embedding which has to be in order to computer attention score
            # Use repeat_kv function to make Keys,Values shape same as queries shape
            keys = repeat_kv(keys, self.n_rep)      #keys[bsz,seq_len,n_heads,head_dim]
            values = repeat_kv(values, self.n_rep)  #values[bsz,seq_len,n_heads,head_dim]

        # Mode - Training mode: KV-Cache not implemented
        else:
            # Compute rotation matrix and apply RoPE to queries and keys for for training.
            freqs_cis = precompute_freqs_cis(dim=self.head_dim, end=self.args.max_seq_len)

            #xq[bsz,seq_len,n_heads, head_dim], xk[bsz,seq_len,n_heads, head_dim]
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

            # Use repeat_kv function to make Keys,Values shape same as the queries shape
            #keys[bsz,seq_len,n_heads,head_dim], #values[bsz,seq_len,n_heads,head_dim]
            keys = repeat_kv(xk, self.n_rep)
            values = repeat_kv(xv, self.n_rep)

            # For training mode, we'll compute mask and apply to the attention score later
            mask = torch.full((seq_len, seq_len),float("-inf"),device=self.args.device)
            mask = torch.triu(mask, diagonal=1).to(self.args.device)

            # To compute attention, we'll need to perform a transpose operation to reshape all queries, keys and values bring heads at dim 1 and seq at dim 2
            xq = xq.transpose(1,2)                  #xq[bsz,n_heads,seq_len,head_dim]
            keys = keys.transpose(1,2)              #keys[bsz,n_heads,seq_len,head_dim]
            values = values.transpose(1,2)          #values[bsz,n_heads,seq_len,head_dim]

            # Computing attention score
            scores = torch.matmul(xq, keys.transpose(2,3)).to(self.args.device)/math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask

            # Apply softmax to the attention score
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            # Matrix multiplication of attention score with the values
            output = torch.matmul(scores, values).to(self.args.device)

            # We get the contextual embedding for each head
            # All heads need to be reshaped back and combined to give a single single contextual attention output
            # Shape change: output[bsz,n_heads,seq_len,head_dim] -> output[bsz,seq_len, n_heads,head_dim] -> output[bsz,seq_len, n_heads * head_dim]
            output = output.transpose(1,2).contiguous().view(bsz, seq_len, -1)

        # shape: output [bsz,seq_len,dim]
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        # Adjust hidden dimension
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # Replace ColumnParallelLinear and RowParallelLinear with nn.Linear
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        # Apply activation and multiply outputs
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.ffn = FeedForward(
            dim=args.dim,
            hidden_dim=args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_mult,
        )
        self.layer_id = layer_id
        self.attn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
    
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        inference
    ):
        h = x + self.attention(self.attn_norm(x), start_pos, inference)
        out = h + self.ffn(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(
            params.vocab_size, params.dim
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        # Replace ColumnParallelLinear with nn.Linear
        self.output = nn.Linear(
            params.dim, params.vocab_size, bias=False
        )

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

    def forward(self, tokens: torch.Tensor, start_pos: int, targets=None, inference=False):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)

            # Adjust mask for key-value caching
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, inference)
        h = self.norm(h)
        logits = self.output(h).float()


        if targets is None:
            loss = None
        # Training mode is activated if the targets are available. And Loss will be calculated for further model training. 
        else:
            loss = F.cross_entropy(logits.view(-1, self.params.vocab_size), targets.view(-1))

        return logits, loss