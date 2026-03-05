# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qwen3.5-MoE model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Qwen3_5_MoeVisionConfig(PretrainedConfig):
    model_type = "qwen3_5_moe"

    def __init__(
        self,
        depth=27,
        hidden_size=1152,
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=4304,
        in_channels=3,
        num_heads=16,
        out_hidden_size=2048,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        num_position_embeddings=2304,
        initializer_range=0.02,
        deepstack_visual_indexes=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.out_hidden_size = out_hidden_size
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.num_position_embeddings = num_position_embeddings
        self.initializer_range = initializer_range
        self.deepstack_visual_indexes = (
            deepstack_visual_indexes
            if deepstack_visual_indexes is not None
            else []
        )


class Qwen3_5_MoeTextConfig(PretrainedConfig):
    model_type = "qwen3_5_moe_text"
    base_config_key = "text_config"

    # FIX: Must be a set, not a list. The transformers library does
    # `ignore_keys_at_rope_validation | {"partial_rotary_factor"}` which
    # requires set union semantics. A list triggers TypeError.
    ignore_keys_at_rope_validation = {
        "partial_rotary_factor",
        "mrope_section",
        "mrope_interleaved",
    }

    def __init__(
        self,
        vocab_size=248320,
        hidden_size=2048,
        num_hidden_layers=40,
        num_attention_heads=16,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=262144,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_parameters=None,
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=256,
        attn_output_gate=True,
        full_attention_interval=4,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        num_experts_per_tok=8,
        num_experts=256,
        norm_topk_prob=True,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        mlp_only_layers=None,
        layer_types=None,
        mtp_num_hidden_layers=1,
        mtp_use_dedicated_embeddings=False,
        mamba_ssm_dtype="float32",
        **kwargs,
    ):
        if mlp_only_layers is None:
            mlp_only_layers = []
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache

        # Handle rope_parameters / rope_scaling
        rope_scaling = kwargs.pop("rope_scaling", None)
        rope_parameters = rope_scaling or rope_parameters or {
            "rope_type": "default"
        }
        rope_theta = kwargs.pop("rope_theta", 10000000.0)
        if "rope_theta" not in rope_parameters:
            rope_parameters["rope_theta"] = rope_theta
        partial_rotary_factor = kwargs.pop("partial_rotary_factor", 0.25)
        if "partial_rotary_factor" not in rope_parameters:
            rope_parameters["partial_rotary_factor"] = partial_rotary_factor
        self.rope_parameters = rope_parameters
        self.partial_rotary_factor = partial_rotary_factor

        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim
        self.attn_output_gate = attn_output_gate
        self.full_attention_interval = full_attention_interval

        # Layer types (alternating linear_attention / full_attention)
        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "linear_attention"
                if bool((i + 1) % full_attention_interval)
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        # Linear attention params
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads

        # MoE params
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.mlp_only_layers = mlp_only_layers

        # MTP (Multi-Token Prediction) params
        self.mtp_num_hidden_layers = mtp_num_hidden_layers
        self.mtp_use_dedicated_embeddings = mtp_use_dedicated_embeddings

        # Mamba params
        self.mamba_ssm_dtype = mamba_ssm_dtype


class Qwen3_5_MoeConfig(PretrainedConfig):
    model_type = "qwen3_5_moe"
    sub_configs = {
        "text_config": Qwen3_5_MoeTextConfig,
        "vision_config": Qwen3_5_MoeVisionConfig,
    }

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=248056,
        video_token_id=248057,
        vision_start_token_id=248053,
        vision_end_token_id=248054,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(text_config, dict):
            text_config = Qwen3_5_MoeTextConfig(**text_config)
        elif text_config is None:
            text_config = Qwen3_5_MoeTextConfig()
        self.text_config = text_config

        if isinstance(vision_config, dict):
            vision_config = Qwen3_5_MoeVisionConfig(**vision_config)
        elif vision_config is None:
            vision_config = Qwen3_5_MoeVisionConfig()
        self.vision_config = vision_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id

    def get_text_config(self, decoder=False):
        return self.text_config


__all__ = [
    "Qwen3_5_MoeConfig",
    "Qwen3_5_MoeTextConfig",
    "Qwen3_5_MoeVisionConfig",
]
