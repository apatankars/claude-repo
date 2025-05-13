from __future__ import annotations
"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

# For assignment #2, we've preset some flags that increase numerical determinism. Control them with this flag!
DEBUG = False

import math
import inspect
from dataclasses import dataclass
from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    def __init__(self, ndim: Sequence[int] | int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias) # hello...? how does this work :)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and not DEBUG
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))
        
        ### Initialize KV cache with up to a maximum sequence length.
        ### Keep track of how much of it is filled in self.seq_pos.
        self.seq_pos = 0 # disabled by default
        self.kv_enabled = False # disabled by default

        # TODO (Part 3): Implement!
        # Initialize a register buffer to store the KV cache; you'll need to enter its dimensions below
        # Hint: it has 5 total dimensions, the first of which is '2', because we need to store 
        # both the key and the value for each token in the sequence
        
        batch_dim = 32 # the maximum batch dimension we'll support. This is because T4 has smol mem.
        seq_dim = 512 # the maximum sequence length we'll support

        # We want the KV cache for every batch. Each head will progessively fill the cache up to the seqence length, and each head is operating on
        # a subsect of the embeddings
        self.register_buffer("kv_cache", torch.empty(2, batch_dim, self.n_head, seq_dim, self.n_embd // self.n_head, dtype=torch.bfloat16), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass for scaled dot-product self-attention, with multiple
        heads of attention.
        
        Args:
            x (torch.Tensor): input activations with shape (batch size, 
            sequence length, embedding dimension)
        
        Returns:
            torch.Tensor: the output of the attention layer
        """
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd) (is sequence_length always the same in forward?)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        ### Set KV cache if it exists:
        if self.kv_enabled:
            # TODO (Part 3): Implement!
            #  1. Set the new position that we are in the KV cache (see `self.seq_pos`)
            #  2. Update the KV cache (given by `self.kv_cache`)
            #  3. Update the KV cache with the new values
            #  4. Update the position we are in the KV cache

            # We update the batches UP TO the batch_num we are on
            # Update all of the attention heads that are predicted in this pass
            # Update the current sequence of tokens we just predicted so (pos : pos + T)?
            # obviously update all the values of the dimensions

            self.kv_cache[0, :B, :, self.seq_pos:self.seq_pos+T, :] = k # (K, B, nh, new_tokens, hs)
            self.kv_cache[1, :B, :, self.seq_pos:self.seq_pos+T, :] = v # (V, B, nh, new_tokens, hs)

            # Need to increment the position in the cache
            self.seq_pos += T

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass for a decoder model. Similar to `forward`, but assumes 
        that `x` contains the tokens appearing after those currently stored within 
        the KV cache.
        
        Args:
            x (torch.Tensor): input activations with shape (batch size, 
            sequence length, embedding dimension)
        
        Returns:
            torch.Tensor: the output of the attention layer
        """
        if not self.kv_enabled:
            raise ValueError("KV cache is not enabled!")
        
        B, T, C = x.size()  # T is always 1 now

        # 1. Compute Q, K, and V projections of `x`
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, 1, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, 1, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, 1, hs)

        #  2. Update the KV cache with the input
        # first we want to update the cache with the new value we just calculated for the new token(s) we just geenerated
        self.kv_cache[0, :B, :, self.seq_pos:self.seq_pos+T, :] = k  # Update keys
        self.kv_cache[1, :B, :, self.seq_pos:self.seq_pos+T, :] = v  # Update values
        
        # now we want to use updated cache values up to this point since it will contain all previous K and V we need
        # as we can take all of the generation up to where we are in the sequence, (seq_pos + T)
        k_full = self.kv_cache[0, :B, :, :self.seq_pos+T, :]  # (B, nh, seq_pos+T, hs)
        v_full = self.kv_cache[1, :B, :, :self.seq_pos+T, :]  # (B, nh, seq_pos+T, hs)

        # 3. Compute manual implementation of attention using the KV cache
        if self.flash:

            y = torch.nn.functional.scaled_dot_product_attention(
                q, k_full, v_full, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
            )
        else:
            # manual implementation of attention
            att = (q @ k_full.transpose(-2, -1)) * (1.0 / math.sqrt(k_full.size(-1)))

             # Create causal mask that allows attention to all previous tokens
            # The shape is (T, seq_pos+T)
            causal_mask = torch.ones((1, 1, T, self.seq_pos+T), device=x.device)
            for i in range(T):
                # For each query position i, allow attention to all previous cached tokens (0:seq_pos)
                # and all current tokens up to position i
                causal_mask[0, 0, i, :self.seq_pos+i+1] = 0
            
            att = att.masked_fill(causal_mask == 1, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v_full # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        self.seq_pos += T  # Update the position in the KV cache
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn.decode(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless.
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    # Some scaffolding from the 1390 teaching staff.
    @property # this is just a convenience function to make accessing easier
    def seq_pos(self) -> int:
        return self.transformer.h[0].attn.seq_pos
    
    @seq_pos.setter
    def seq_pos(self, value: int | None):
        for block in self.transformer.h:
            block.attn.seq_pos = value
    
    def enable_kv(self, use_kv: bool = True):
        self.seq_pos = 0 if use_kv else None
        self.kv_enabled = use_kv
        for block in self.transformer.h:
            block.attn.kv_enabled = use_kv

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Performs the forward pass of the GPT model.
        
        Args:
            idx (torch.Tensor): input sequence of vocab indices with shape (batch size, sequence length)
            targets (torch.Tensor | None): the optional desired targets of the 
            model, to calculate loss against
        
        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: the output of the model and
            optionally the calculated loss against the given targets, if specified
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # this first is creating an array marking each position in the sequence with in index essentially
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb) # create the combined embeddings using drop?
        for block in self.transformer.h: # iterate over all of the transformer hidden blocks and call them
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x)
            loss = None

        return logits, loss
    
    def decode(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the GPT decoder model. Similar to `forward`, but assumes 
        that `idx` contains the tokens appearing after those currently stored within 
        the model's KV caches.
        
        Args:
            idx (torch.Tensor): input sequence of vocab indices (token IDs) with shape (batch size, 
            sequence length)
        Returns:
            torch.Tensor: the output of the model
        """
        device = idx.device
        b, t = idx.size()
        assert self.seq_pos + t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # 1. Create token and position embeddings (from above)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        
        # Position embeddings need to start from self.seq_pos
        # (similar to above, but now we only want the positions from the current point in the sequence to the tokens we generated 
        pos = torch.arange(self.seq_pos, self.seq_pos + t, dtype=torch.long, device=device)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        
        x = self.transformer.drop(tok_emb + pos_emb) # now we can combine them as above using drop
        
        # 2. Perform decoding using the KV caches
        for block in self.transformer.h: # iterate over the blocks in the transformer
            x = block.decode(x)  # Use the block's decode method which will use the KV cache
        
        # 3. Apply transformer layer normalization
        x = self.transformer.ln_f(x)
        
        # 4. Apply the language model head and return the output
        logits = self.lm_head(x)

        return logits

    def crop_block_size(self, block_size: int):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type: str, override_args: dict | None = None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay: float, learning_rate: float, 
                             betas: tuple[float, float], device_type: str):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float):
        """Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, 
                 temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        """
        Receives a conditioning sequence of tokens and predicts the following tokens 
        in the sequence, feeding the predictions back into the model each time.
        This function should be used in model evaluation mode (`model.eval()`).
        
        Args:
            idx (torch.Tensor): the conditioning sequence of tokens, of shape 
            (batch size, sequence length)
            max_new_tokens (int): maximum number of new tokens to predict
            temperature (float): a scaling factor in [0, 1] for token generation, 
            such that values close to 1 make the output distribution more uniform (random), 
            while values close to 0 make it more concentrated; defaults to 1
            top_k (int | None): limits the token sampling to the top `k` 
            highest-probability tokens if specified, otherwise, no restriction is applied;
            defaults to None
        
        Returns:
            torch.Tensor: the extended sequence with the generated predictions
            concatenated on the end of the input sequence
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                if top_k == 1:
                    idx = torch.cat((idx, torch.argmax(logits, dim=-1).reshape((-1,1))), dim=1) # using argmax here preserves RNG state
                    continue
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)

            # sample from the distribution with temperature
            main_pred = torch.multinomial(probs, num_samples=1)

            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, main_pred), dim=1)

        return idx

    @torch.no_grad()
    def generate_kv(self, idx: torch.Tensor, max_new_tokens: int, 
                    temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        """
        Receives a conditioning sequence of tokens and predicts the following tokens 
        in the sequence, feeding the predictions back into the model each time; 
        similar to `generate`, though uses KV caching.
        
        Args:
            idx (torch.Tensor): the conditioning sequence of tokens, of shape 
            (batch size, sequence length)
            max_new_tokens (int): maximum number of new tokens to predict
            temperature (float): a scaling factor in [0, 1] for token generation, 
            such that values close to 1 make the output distribution more uniform (random), 
            while values close to 0 make it more concentrated; defaults to 1
            top_k (int | None): limits the token sampling to the top `k` 
            highest-probability tokens if specified, otherwise, no restriction is applied;
            defaults to None
        
        Returns:
            torch.Tensor: the extended sequence with the generated predictions
            concatenated on the end of the input sequence
        """

        # TODO gameplan:
        # 1). Need to call forward on the model once to initalize the cache and get generation started
        # 2). Properly create a dyanmic tensor to keep track of the growing sequence
        # 3). Generate and append additional tokens using decode, where only pass in the most recently generated token
        # 4). Generate tokens until the max sequence length (are there other stopping conditions?)
        
        # Generate the full KV cache and decode the first next token
        self.enable_kv(True)

        # Start with the input sequence
        b, t = idx.size()

        # 1. Run `idx` through the forward pass to get the logits
        logits, _ = self(idx) # i think we just call self? (B, N, D)

        # this is a buffer to which we are going to append our new tokens to (we need for part 2)
        # note: needed to explicitly cast this to a long and make sure it was on the same device as the input
        new_tokens = torch.empty((b, 0), dtype=torch.long, device=idx.device) 

        # 2. Pluck the logits at the final position and scale by desired temperature
        logits = logits[:, -1, :] / temperature # select the final token from the sequence dimension (2) for all batches and dimensions

        # 3. Optionally crop the logits to only the top k options, as specified (similar to generate above)
        # are we allowed to assume that top_k is always non-null for this?
        if top_k == 1:
            main_pred = torch.argmax(logits, dim=-1).reshape((-1,1)) # we reshape to be of the proper shape (B, 1)
        else:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1))) # sample the top-K values 
            logits[logits < v[:, [-1]]] = -float('Inf') # mask all tokens that have a prob < lowest top-K prob to -inf
            probs = F.softmax(logits, dim=-1) # renormalize the distribution so now the top-k are redistributed
            main_pred = torch.multinomial(probs, num_samples=1) # sample from the normalized top-K distribution

        # 4. Save the decoded next token to `new_tokens`
        new_tokens = torch.cat((new_tokens, main_pred), dim=1)

        for _ in range(max_new_tokens-1):
            # Now decode using the KV cache

            # 1. Decode from the new tokens to get the logits
            # Get single token to be decoded
            ntok = new_tokens[:, -1:]
            
            # Decode the next token using KV cache
            logits = self.decode(ntok)

            # 2. Pluck the logits at the final position and scale by desired temperature
            logits = logits[:, -1, :] / temperature

            # If top-k sampling is enabled, then sample the top-K values from the logits, and mask all values below the lowest top-K prob.
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1))) # this is the sampling of the top-K tokens
                logits[logits < v[:, [-1]]] = -float('Inf') # this masks all logits < lowest top-k prob to -inf

            # If top_k and top_k == 1, then we just take the top entry            
            if top_k == 1:
                main_pred = torch.argmax(logits, dim=-1, keepdim=True)
            else: # Any other type, we sample from the distribution (top-k alters the underlying distribution)
                probs = F.softmax(logits, dim=-1)
                main_pred = torch.multinomial(probs, num_samples=1)

            # 4. Append decoded next token to `new_tokens`
            new_tokens = torch.cat((new_tokens, main_pred), dim=1)

        return torch.cat((idx, new_tokens), dim=1)

    @torch.no_grad()
    def generate_speculative(self, idx: torch.Tensor, max_new_tokens: int, 
                             draft_model: GPT, temperature: float = 1.0, 
                             top_k: int | None = None, num_speculative: int = 4) -> torch.Tensor:
        """
        Receives a conditioning sequence of tokens and predicts the following tokens 
        in the sequence, feeding the predictions back into the model each time; 
        similar to `generate`, though uses speculative decoding.
        
        Args:
            idx (torch.Tensor): the conditioning sequence of tokens, of shape 
            (batch size, sequence length)
            max_new_tokens (int): maximum number of new tokens to predict
            draft_model (GPT): draft model to speculative decode with
            temperature (float): a scaling factor in [0, 1] for token generation, 
            such that values close to 1 make the output distribution more uniform (random), 
            while values close to 0 make it more concentrated; defaults to 1
            top_k (int | None): limits the token sampling to the top `k` 
            highest-probability tokens if specified, otherwise, no restriction is applied;
            defaults to None
            num_speculative (int): number of speculative tokens to generate using
            draft model; defaults to 4
        
        Returns:
            torch.Tensor: the extended sequence with the generated predictions
            concatenated on the end of the input sequence
        """

        self.enable_kv(False) # disable KV cache for the main model -- it's not worth the effort
        draft_model.enable_kv(False)

        # Note: making speculative decoding work with batch_size > 1 is beyond this assignment's scope, because the
        # tensors rapidly become ragged. So, you can assume that batch_size = 1 for this part.
        if idx.size(0) != 1:
            raise ValueError("Speculative decoding only works with batch size 1")
        idx_length_original = idx.size(1)
        
        loop_counter = 0
        while idx.size(1) < idx_length_original+max_new_tokens: # keep going until sequence is at max_length
            loop_counter += 1

            # if the sequence context is growing too long we must crop it at block_size (why??)
            idx_cond = idx if idx.size(1) <= self.config.block_size-num_speculative else idx[:, -self.config.block_size-num_speculative:]

            # 1. Use the draft_model to generate speculative tokens (top-k is always 1)

            # Maybe clone the idx because we don't want the draft model to modify it possibly?
            draft_pred = idx_cond.clone() 

            # we want the smaller model to generate multiple tokens at a time, so loop through num speculative tokens
            for _ in range(num_speculative):
                # Generate next token with draft model (greedy decoding)
                draft_logits, _ = draft_model(draft_pred)
                draft_logits = draft_logits[:, -1, :] / temperature
                ntok = torch.argmax(draft_logits, dim=-1, keepdim=True) # since top-k is always 1, we can just use argmax
                draft_pred = torch.cat((draft_pred, ntok), dim=1) 

            # 2. Obtain the logits from the main model by passing in the draft_pred
            main_logits, _ = self(draft_pred) # this is the output of the model given the predicted draft predicted seqeunce
            
            accepted_count = 0
            init_seq_len = idx_cond.size(1) # initial length of the sequence before generation
            pred_seq_len = draft_pred.size(1)

            # Step through the predictions of the main model, sampling, and check whether they match the next token. Stop upon mismatch.
            # 3. Iterate from the end position of idx_cond (prefix sequence) to the end position of draft_pred (generated sequence)
            for i in range(init_seq_len, draft_pred.size(1)): # iterate through the new generated tokens

                # 4. pluck the logits at the current position and scale by desired temperature (same as above)
                logits = main_logits[:, i-1, :] / temperature # get the token generated by the larger model for this timestep (i-1)
                
                # 5. optionally crop the logits to only the top k options (slightly altered now since we only care about the specifc token)
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                # 6. Apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                
                # 7. Sample from the distribution with temperature
                if top_k == 1:
                    main_pred = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    main_pred = torch.multinomial(probs, num_samples=1)
                
                draft_speculative_token = draft_pred[:, i].view(1, 1).item()
                main_speculative_token = main_pred.item()
                
                if main_speculative_token != draft_speculative_token:
                    # Mismatch case

                    # If we have accepted at least some tokens so far, we will append them to the final 
                    if accepted_count > 0:
                        # properly append the new tokens from the draft model to the running sequence
                        idx = torch.cat((idx, draft_pred[:, init_seq_len:init_seq_len+accepted_count]), dim=1)

                    # we append the correctly predicted token by the main model so we continue making progress and break from the loop
                    idx = torch.cat((idx, main_pred), dim=1) 
                    break
                else:
                    # Match case

                    # we increment the token counter
                    accepted_count += 1
                    
                    # If we read the end of the predicted sequence, all token are right so we add them all
                    if i == pred_seq_len - 1:
                        idx = torch.cat((idx, draft_pred[:, init_seq_len:]), dim=1)
                
        print(f"Speculative decoding ran for {loop_counter} iterations")
        return idx[:,:idx_length_original+max_new_tokens]
