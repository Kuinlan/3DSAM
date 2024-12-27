import torch
import torch.nn as nn
import math

class PerceiverSelfAttention(nn.Module):
    """Perceiver Self-Attention module, used as part of the Perceiver IO model."""
    def __init__(
        self,
        is_cross_attention,
        qk_channels,
        v_channels,
        num_heads,
        q_dim,
        kv_dim,
        attention_dropout,
    ):
        super().__init__()

        self.num_heads = num_heads

        if qk_channels is None:
            qk_channels = q_dim
        if v_channels is None:
            v_channels = qk_channels

        if qk_channels % num_heads != 0:
            raise ValueError(f"qk_channels ({qk_channels}) must be divisible by num_heads ({num_heads}).")
        if v_channels % num_heads != 0:
            raise ValueError(f"v_channels ({v_channels}) must be divisible by num_heads ({num_heads}).")

        self.qk_channels = qk_channels
        self.v_channels = v_channels
        self.qk_channels_per_head = self.qk_channels // num_heads
        self.v_channels_per_head = self.v_channels // num_heads

        self.layernorm1 = nn.LayerNorm(q_dim)
        self.layernorm2 = nn.LayerNorm(kv_dim) if is_cross_attention else nn.Identity()

        # Key/query/value projections
        self.query = nn.Linear(q_dim, qk_channels)
        self.key = nn.Linear(kv_dim, qk_channels)
        self.value = nn.Linear(kv_dim, v_channels)

        # Optional dropout
        self.dropout = nn.Dropout(attention_dropout)

    def transpose_for_scores(self, x, channels_per_head):
        """Reshape input for self-attention"""
        new_x_shape = x.size()[:-1] + (self.num_heads, channels_per_head)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        inputs=None,
        inputs_mask=None,
    ):
        """ Forward pass of the Perceiver Self-Attention module"""
        hidden_states = self.layernorm1(hidden_states)
        inputs = self.layernorm2(inputs)

        is_cross_attention = inputs is not None
        queries = self.query(hidden_states)

        if is_cross_attention:
            keys = self.key(inputs)
            values = self.value(inputs)
            attention_mask = inputs_mask
        else:
            keys = self.key(hidden_states)
            values = self.value(hidden_states)

        queries = self.transpose_for_scores(queries, self.qk_channels_per_head)
        keys = self.transpose_for_scores(keys, self.qk_channels_per_head)
        values = self.transpose_for_scores(values, self.v_channels_per_head)

        attention_scores = torch.matmul(queries, keys.transpose(-1, -2))

        _, _, _, q_head_dim = queries.shape
        _, _, _, v_head_dim = values.shape
        hidden = self.num_heads * v_head_dim

        attention_scores = attention_scores / math.sqrt(q_head_dim)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, values)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (hidden,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return {
            'context': context_layer,
            'attention_prob': attention_probs,
        }


class PerceiverSelfOutput(nn.Module):
    """Simple class wrapping up a linear layer."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dense = nn.Linear(in_channels, out_channels)

    def forward(self, hidden_state):
        """Forward pass of the PerceiverSelfOutput"""
        return self.dense(hidden_state)

class PerceiverAttention(nn.Module):
    """Perceiver attention module, used for self-attention and cross-attention."""
    def __init__(
        self,
        is_cross_attention,
        qk_channels,
        v_channels,
        num_heads,
        q_dim,
        kv_dim,
        use_query_residual,
        cross_attention_shape,
        attention_dropout,
        use_flash_attention,
    ):
        super().__init__()
        self.use_query_residual = use_query_residual
        self.is_cross_attention = is_cross_attention
        self.with_flash_attention = use_flash_attention

        if is_cross_attention and qk_channels is None:
            if cross_attention_shape == "q":
                qk_channels = q_dim
            elif cross_attention_shape == "kv":
                qk_channels = kv_dim
            else:
                raise ValueError(
                    f"Unknown value {cross_attention_shape} for "
                    "cross_attention_shape_for_attention."
                )
        else:
            if qk_channels is None:
                qk_channels = q_dim
            if v_channels is None:
                v_channels = qk_channels

        self.self = PerceiverSelfAttention(
            is_cross_attention=is_cross_attention,
            qk_channels=qk_channels, v_channels=v_channels,
            num_heads=num_heads, q_dim=q_dim, kv_dim=kv_dim,
            attention_dropout=attention_dropout,
        )
        self.output = PerceiverSelfOutput(
            in_channels=self.self.v_channels,
            out_channels=q_dim if is_cross_attention else v_channels
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        inputs=None,
        inputs_mask=None,
    ):
        """Forward pass for the attention module, returns attention values and projection values"""
        if not self.is_cross_attention and self.with_flash_attention:
            attention_output = self.self(hidden_states)[0]
            if self.use_query_residual:
                attention_output = attention_output + hidden_states
            return {
                'mlp': attention_output,
            }

        self_output = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            inputs,
            inputs_mask,
        )

        attention_output = self.output(self_output['context'])  # 1024 -> 132
        if self.use_query_residual:
            attention_output = attention_output + hidden_states

        return {
            'attention': self_output,
            'mlp': attention_output,
        }


class PerceiverMLP(nn.Module):
    """A Transformer-style dense module to follow attention."""

    def __init__(self, hidden_activ, input_size, widening_factor):
        super().__init__()
        self.dense1 = nn.Linear(input_size, widening_factor * input_size)
        if isinstance(hidden_activ, str):
            self.intermediate_act_fn = nn.functional.gelu
        else:
            self.intermediate_act_fn = hidden_activ
        self.dense2 = nn.Linear(input_size, input_size)

    def forward(self, hidden_states):
        """Forward pass of the Perceiver MLP"""
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense2(hidden_states)
        return hidden_states       



class PerceiverLayer(nn.Module):
    """Perceiver Layer with attention and MLP"""
    def __init__(
        self,
        is_cross_attention,
        use_query_residual,
        qk_channels,
        v_channels,
        num_heads,
        q_dim,
        kv_dim,
        widening_factor,
        hidden_activ,
        attention_dropout,
        use_flash_attention,
        cross_attention_shape=None,
    ):
        super().__init__()
        self.attention = PerceiverAttention(
            is_cross_attention=is_cross_attention,
            qk_channels=qk_channels,
            v_channels=v_channels,
            num_heads=num_heads,
            q_dim=q_dim,
            kv_dim=kv_dim,
            use_query_residual=use_query_residual,
            cross_attention_shape=cross_attention_shape,
            attention_dropout=attention_dropout,
            use_flash_attention=use_flash_attention,
        )
        self.layernorm = nn.LayerNorm(q_dim)
        self.mlp = PerceiverMLP(
            input_size=q_dim,
            hidden_activ=hidden_activ,
            widening_factor=widening_factor
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        inputs=None,
        inputs_mask=None,
    ):
        """ Forward pass of the Perceiver Layer"""

        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            inputs,
            inputs_mask,
        )

        attention_output = attention_outputs['mlp']  # 132

        context = attention_outputs['attention']['context']  # 1024
        
        mlp_output = self.layernorm(attention_output)
        mlp_output = self.mlp(mlp_output)
        mlp_output = mlp_output + attention_output

        return {
            'context': context,
            'attention': attention_output,
            'mlp': mlp_output
        }