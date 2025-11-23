# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Any, Dict, List, Optional

import torch

from sglang.srt.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from sglang.srt.layers.parameter import GroupQuantScaleParameter, PackedvLLMParameter
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.utils import is_cuda



logger = logging.getLogger(__name__)



class MixQ8Config(QuantizationConfig):
    """Config class for MixQ4.

    Reference: https://arxiv.org/abs/2306.00978
    """

    def __init__(
        self,
        weight_bits,
        group_size,
        zero_point,
        modules_to_not_convert
    ) -> None:
        super().__init__()
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.modules_to_not_convert = modules_to_not_convert or []
        if self.weight_bits != 8:
            raise ValueError(
                "Currently, only 8-bit weight quantization is supported"
            )


    def __repr__(self) -> str:
        return (
            f"MixQConfig(weight_bits={8}, "
            f"group_size={self.group_size}, "
            f"zero_point={self.zero_point}, "
            f"modules_to_not_convert={self.modules_to_not_convert})"
        )
    def get_name(self) -> str:
        return "mixq8"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16, torch.float8_e4m3]

    @classmethod
    def get_min_capability(cls) -> int:
        # The AWQ kernel only supports Turing or newer GPUs.
        return 80

    @staticmethod
    def get_config_filenames() -> List[str]:
        return [
            "quant_config.json",
            ]
    def get_scaled_act_names(self) -> List[str]:
            return []
    @classmethod
    def from_config(cls, config: Dict[str, Any]) :
        weight_bits = 8
        group_size = cls.get_from_keys(config, ["q_group_size", "group_size"])
        zero_point = False
        modules_to_not_convert = cls.get_from_keys_or(
            config, ["modules_to_not_convert"], None
        )
        return cls(weight_bits, group_size, zero_point, modules_to_not_convert)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str) -> Optional["LinearMethodBase"]:
        def is_layer_skipped_(prefix: str, modules_to_not_convert: List[str]):
            return any(module_name in prefix for module_name in modules_to_not_convert)
        if isinstance(layer, LinearBase):
            if is_layer_skipped_(prefix, self.modules_to_not_convert):
                return UnquantizedLinearMethod()
            return MixQ8LinearMethod(self)
        return None


class MixQ8LinearMethod(LinearMethodBase):


    def __init__(self, quant_config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):



        output_size_per_partition = sum(output_partition_sizes)


        weight_loader = extra_weight_attrs.get("weight_loader")
        q_weight = PackedvLLMParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.int8,
            ),
            input_dim=1,
            output_dim=0,
            packed_dim=1,
            packed_factor=1,
            weight_loader=weight_loader,
        )

        q_scale_col = GroupQuantScaleParameter(
            data=torch.empty(
                1,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=0,
            output_dim=1,
            weight_loader=weight_loader,
        )

 

        layer.register_parameter("q_weight", q_weight)
        layer.register_parameter("q_scale_col", q_scale_col)


    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.q_weight = torch.nn.Parameter(layer.q_weight.data, requires_grad=False)
        layer.q_scale_col = torch.nn.Parameter(layer.q_scale_col.data, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:


        w = layer.q_weight.to(torch.float16) * layer.q_scale_col.to(torch.float16).T

        out = torch.mm(x, w.T)
        if bias is not None:
            out.add_(bias)
        return out
