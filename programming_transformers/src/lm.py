import math
from collections import namedtuple
from collections.abc import Callable
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth

class GPT2Embeddings(nn.Module):
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        max_position_embeddings: int,
        padding_idx: int | None = None,
        word_embed_proj_dim: int | None = None,
        dtype: str = "torch.float32",
    ):
        """
        If max_position_embeddings <= 0, there's no position embeddings
        We embed to word_embed_proj_dim dimension then project up to d_model
        """
        super().__init__()
        self.dtype = dtype
        if word_embed_proj_dim is None:
            self.word_embeddings = nn.Embedding(
                vocab_size, d_model, padding_idx=padding_idx
            )
            self.project_in = None
        else:
            self.word_embeddings = nn.Embedding(
                vocab_size,
                word_embed_proj_dim,
                padding_idx=padding_idx,
            )
            self.project_in = nn.Linear(word_embed_proj_dim, d_model, bias=False)
        self.max_position_embeddings = max_position_embeddings
        if self.max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(max_position_embeddings, d_model)

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        """
        input_ids: (batch, seqlen)
        position_ids: (batch, seqlen)
        """
        batch_size, seqlen = input_ids.shape
        embeddings = self.word_embeddings(input_ids)
        if self.project_in is not None:
            embeddings = self.project_in(embeddings)
        if self.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = torch.arange(
                    seqlen, dtype=torch.long, device=input_ids.device
                )
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
        return embeddings


def _init_weights(
    module: nn.Module,
    n_layers: int,
    initializer_range: float = 0.02,
    rescale_prenorm_residual: bool = True,
):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                nn.init.normal_(
                    p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layers)
                )
            # If using GLU activation for now, we scale the std by 2
            elif name in ["output_linear.0.weight"]:
                nn.init.normal_(
                    p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layers)
                )


class ScaledDotProdAttention(nn.Module):
    """
    Module implementation of masked scaled dot-product attention.
    
    Attributes:
        dropout_p (float): the dropout probability to apply during attention when training 
    """
    def __init__(self, attention_dropout: float = 0.0):
        """
        Initializes a masked scaled dot-product attention.
        
        Args:
            attention_dropout (float): the dropout probability to apply during 
            attention when training
        """
        super().__init__()
        self.dropout_p = attention_dropout

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        """
        Implements masked scaled dot-product attention.

        Args:
            qkv (torch.Tensor): the tensor containing the queries, keys, and values,
            with a shape (B, S, 3, H, d); see `MHA` for a explanation of shape conventions
        Returns:
            torch.Tensor: the result of masked scaled dot-product attention, with 
            shape (B, S, H, d)
        """
        batch_size, seqlen = qkv.shape[0], qkv.shape[1] # correspond to B, S
        q, k, v = qkv.unbind(dim=2) # q, k, v each have a shape of (B, S, H, d)

        batch_size, seq_len, num_heads, head_dim = q.size()

        # TODO (Part 1.1): Implement!
        softmax_scale = 1.0 / math.sqrt(head_dim)
        dot_product_scores = torch.einsum("bshd,btjd->bsht", q, k) * softmax_scale # (B, S, H, T) since now we want every token in our query (S) to attend to every token in the key (T)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device), diagonal=1) # (S, S)
        dot_product_scores_masked = dot_product_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(2), float("-inf")) # need to unsqueeze to (1, S, 1, S) to match the shape of dot_product_scores
        attention = F.softmax(dot_product_scores_masked, dim=-1) # (B, S, H, S) applied to the last dimension of dot_product_scores_masked

        attention_drop = F.dropout(attention, self.dropout_p, self.training)

        output = torch.einsum("bsht,btjd->bshd", attention_drop, v) # (B, S, H, d)

        return output


class MHA(nn.Module):
    """
    Module implementation of masked multi-head self-attention.
    Generates queries, keys, and values, performs the attention calculations
    (see `ScaledDotProdAttention` class) and then the output projection.
    
    Attributes:
        d_model (int): the hidden dimension/embedding size
        num_heads (int): number of heads (parallel computations of attention)
        head_dim (int): head dimension/embedding size per head
        attn (ScaledDotProdAttention): module implementing masked multi-head self-attention

    Shape conventions:
        B: batch size
        S: sequence length
        H: number of heads (parallel computations of attention)
           stored as self.num_heads
        D: hidden dimension/embedding size
           stored as self.d_model
        d: head dimension/embedding size per head = D // H
           stored as self.head_dim
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        """
        Initializes a masked multi-head attention module.
        
        Args:
            d_model: (int): the hidden dimension/embedding size
            num_heads (int): the number of heads (parallel computations of attention);
            defaults to 1
            bias (bool): whether the query, key, and value projection layers are biased;
            defaults to True
            dropout (float): the dropout probability to apply during attention when training
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert self.d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = self.d_model // num_heads

        # TODO (Part 1.1): Create the query, key, and value projection layer(s)
        # See the passed-in argument `bias`.
        self.down_projection = nn.Linear(d_model, 3 * d_model, bias=bias)

        self.up_projection = nn.Linear(d_model, d_model, bias=bias)

        self.attn = ScaledDotProdAttention(attention_dropout=dropout)
        
        # TODO (Part 1.1): Create the output projection layer

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Runs masked self-attention.

        Args:
            x (torch.Tensor): input activations with shape (B, S, D); see above
            for explanation of shape conventions
        Returns:
            torch.Tensor: the output of the attention layer
        """
        # TODO (Part 1.1): Generate the queries, keys, and values. Note the shape 
        # of qkv should be (B, S, 3, H, d), where the queries, keys, and values 
        # should lie along the third dimension (of 3) in that order.

        qkv = self.down_projection(x) # (B, S, 3*D) in one step so we can create Q, K, V in one go

        qkv = qkv.view(x.size(0), x.size(1), 3, self.num_heads, self.head_dim) # We can now split this into (B, S, 3, H, d) since 3*D = 3 * H * d

        attention_output = self.attn(qkv, **kwargs) # (B, S, H, d)

        # TODO (Part 1.1): Compute and return the output projection over `attention_output`
        output = attention_output.view(x.size(0), x.size(1), self.d_model) # First we need to reshape the attention_output to (B, S, H * d)

        return self.up_projection(output) # Then we can apply the up_projection to get the final output


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.gelu,
        return_residual: bool = False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.return_residual = return_residual
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        return y if not self.return_residual else (y, x)


class TransformerMixerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 1,
        norm_cls: type[nn.LayerNorm] = nn.LayerNorm,
        dropout_cls: type[nn.Dropout] = nn.Dropout,
        resid_dropout1: float = 0.1,
        resid_dropout2: float = 0.0,
        drop_path1: float = 0.0,
        drop_path2: float = 0.0,
    ):
        super().__init__()
        self.sequence_mixer = MHA(
            d_model,
            num_heads=num_heads,
            dropout=0.1,
        )
        self.state_mixer = Mlp(
            d_model,
            hidden_features=d_model * 4,
            out_features=d_model,
            activation=torch.tanh,
        )
        self.dropout1 = dropout_cls(resid_dropout1)
        self.drop_path1 = StochasticDepth(drop_path1, mode="row")
        self.norm1 = norm_cls(d_model)
        self.dropout2 = dropout_cls(resid_dropout2)
        self.drop_path2 = StochasticDepth(drop_path2, mode="row")
        self.norm2 = norm_cls(d_model)

    def forward(self, hidden_states: torch.Tensor, 
                residual: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        dropped: torch.Tensor = self.drop_path1(self.dropout1(hidden_states))
        residual = (dropped + residual) if residual is not None else dropped
        hidden_states = self.norm1(residual.to(dtype=self.norm1.weight.dtype))
        hidden_states = self.sequence_mixer(hidden_states)

        dropped: torch.Tensor = self.drop_path2(self.dropout2(hidden_states))
        residual = (dropped + residual) if residual is not None else dropped
        hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
        hidden_states = self.state_mixer(hidden_states)
        return hidden_states, residual


class LMBackbone(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        n_layers: int = 12,
        vocab_size: int = 50257,
        num_heads: int = 12,
        max_position_embeddings: int = 0,
        resid_dropout: float = 0.0,
        embed_dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        **kwargs
    ):
        super().__init__()
        self.embeddings = GPT2Embeddings(d_model, vocab_size, max_position_embeddings)
        self.layers = nn.ModuleList(
            [
                TransformerMixerBlock(
                    d_model,
                    num_heads=num_heads,
                    norm_cls=nn.LayerNorm,
                    dropout_cls=nn.Dropout,
                    resid_dropout1=embed_dropout if i == 0 else resid_dropout,
                    resid_dropout2=resid_dropout,
                )
                for i in range(n_layers)
            ]
        )
        self.drop_f = nn.Dropout(resid_dropout)
        self.ln_f = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.apply(
            partial(
                _init_weights,
                n_layers=n_layers,
            )
        )

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        hidden_states = self.embeddings(
            input_ids,
            position_ids=position_ids,
        )
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
        dropped = self.drop_f(hidden_states)
        residual = (dropped + residual) if residual is not None else dropped
        hidden_states = self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))
        return hidden_states


class LMHeadModel(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        n_layers: int = 12,
        vocab_size: int = 50257,
        num_heads: int = 12,
        max_position_embeddings: int = 0,
        resid_dropout: float = 0.0,
        embed_dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        pad_vocab_size_multiple: int = 1,
        **kwargs
    ):
        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (
                vocab_size % pad_vocab_size_multiple
            )

        self.backbone = LMBackbone(
            d_model=d_model,
            n_layers=n_layers,
            vocab_size=vocab_size,
            num_heads=num_heads,
            max_position_embeddings=max_position_embeddings,
            resid_dropout=resid_dropout,
            embed_dropout=embed_dropout,
            layer_norm_epsilon=layer_norm_epsilon,
            **kwargs,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layers=n_layers,
            )
        )
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.backbone.embeddings.word_embeddings.weight

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None) -> tuple[torch.Tensor]:
        hidden_states = self.backbone(input_ids, position_ids=position_ids)
        lm_logits = self.lm_head(hidden_states)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)
