# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/moe_wna16.py

import logging
from typing import Any, Callable, Dict, List, Optional

import torch

from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.layers.linear import LinearBase, UnquantizedLinearMethod
from sglang.srt.layers.quantization.mixq8 import MixQ8Config
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.mixq4 import MixQ4Config

from sglang.srt.utils import get_device_capability, set_weight_attrs

logger = logging.getLogger(__name__)

import mixgemm
import moe_gemm

class MoeMixQ4Config(QuantizationConfig):
    """Config class for MOE WNA16 (W8A16/W4A16) quantization."""

    def __init__(
        self,
        linear_quant_method: str,
        weight_bits: int,
        group_size: int,
        has_zp: bool,
        lm_head_quantized: bool,
        full_config: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.weight_bits = weight_bits
        self.group_size = group_size
 
 
        self.lm_head_quantized = lm_head_quantized
        self.linear_quant_method = linear_quant_method
        self.full_config = full_config
 

    @classmethod
    def get_name(cls) -> str:
        return "moe_mixq4"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    def get_scaled_act_names(self) -> List[str]:
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        quant_method = cls.get_from_keys(config, ["quant_method"])
        weight_bits = cls.get_from_keys(config, ["w_bit"])
        group_size = cls.get_from_keys(config, ["q_group_size"])
        lm_head_quantized = False

        has_zp = False
        return cls(
            quant_method,
            weight_bits,
            group_size,
            has_zp,
            lm_head_quantized,
            config,
        )

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:
        return cls.get_name()



    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        # avoid circular import
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE


        if isinstance(layer, LinearBase):

            
            return UnquantizedLinearMethod()


        elif isinstance(layer, FusedMoE):

            return MoeMixQ4Method(self)
        
        return None


def is_layer_skipped_quant(prefix: str, modules_to_not_convert: List[str]):
    return any(module_name in prefix for module_name in modules_to_not_convert)


class MoeMixQ4Method:
    """Linear method for MOE WNA16 (W8A16/W4A16) quantization.

    Args:
        quant_config: The MOE WNA16 (W8A16/W4A16) quantization config.
    """

    def __new__(cls, *args, **kwargs):
        # avoid circular import
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoEMethodBase

        if not hasattr(cls, "_initialized"):
            original_init = cls.__init__
            new_cls = type(
                cls.__name__,
                (FusedMoEMethodBase,),
                {
                    "__init__": original_init,
                    **{k: v for k, v in cls.__dict__.items() if k != "__dict__"},
                },
            )
            obj = super(new_cls, new_cls).__new__(new_cls)
            obj.__init__(*args, **kwargs)
            return obj
        return super().__new__(cls)

    def __init__(self, quant_config: MoeMixQ4Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        print("create moe mixq4 weights")

        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        layer.quant_config = self.quant_config
        group_size = self.quant_config.group_size
        group_size_div_factor = 1

        # make intermediate_size and hidden_size diviable by group_size
        # we reduce the group size to ensure that
        # and we would repeat the loaded_weight later
        while intermediate_size_per_partition % group_size or hidden_size % group_size:
            group_size = group_size // 2
            group_size_div_factor *= 2
            assert group_size >= 32
        layer.group_size = group_size
        layer.group_size_div_factor = group_size_div_factor
        layer.intermediate_size_per_partition = intermediate_size_per_partition
        layer.hidden_size = hidden_size


        strategy = FusedMoeWeightScaleSupported.GROUP.value
        extra_weight_attrs.update({"quant_method": strategy, "is_transposed": False})

        assert "weight_loader" in extra_weight_attrs
        weight_loader = extra_weight_attrs["weight_loader"]
        wrapped_weight_loader = MoeMixQ4Method.get_weight_loader(layer, weight_loader)
        extra_weight_attrs["weight_loader"] = wrapped_weight_loader

        # Fused gate_up_proj (column parallel)
        w13_q_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 8 ,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_q_weight", w13_q_weight)
        set_weight_attrs(w13_q_weight, extra_weight_attrs)

        w13_q_scale_col = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 128 ,
                dtype=torch.float16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_q_scale_col", w13_q_scale_col)
        set_weight_attrs(w13_q_scale_col, extra_weight_attrs)

        w13_weight_cache = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                128 ,
                dtype=torch.float16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_cache", w13_weight_cache)
        set_weight_attrs(w13_weight_cache, extra_weight_attrs)


        w13_ind = torch.nn.Parameter(
            torch.empty(
                num_experts,
                128,
                1,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_ind", w13_ind)
        set_weight_attrs(w13_ind, extra_weight_attrs)



        w2_weight_cache = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                128,
                dtype=torch.float16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_cache", w2_weight_cache)
        set_weight_attrs(w2_weight_cache, extra_weight_attrs)


        fp_features_num = 128
        import os
        fp = os.getenv("FP_features_num") 
        if fp is not None:
            fp_features_num = fp
        layer.fp_features_num = fp_features_num
        w2_ind = torch.nn.Parameter(
            torch.empty(
                num_experts,
                fp_features_num,
                1,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_ind", w2_ind)
        set_weight_attrs(w2_ind, extra_weight_attrs)

        w2_q_scale_col = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 128 ,
                dtype=torch.float16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_q_scale_col", w2_q_scale_col)
        set_weight_attrs(w2_q_scale_col, extra_weight_attrs)


        w2_q_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 8,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_q_weight", w2_q_weight)
        set_weight_attrs(w2_q_weight, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        num_fused_shared_experts: int = 0,
        custom_routing_function: Optional[Callable] = None,
        correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        inplace: bool = True,
        no_combine: bool = False,
        routed_scaling_factor: Optional[float] = None,
    ) -> torch.Tensor:

        from sglang.srt.layers.moe.topk import select_experts
        assert activation == "silu", "Only SiLU activation is supported."
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            num_fused_shared_experts=num_fused_shared_experts,
            custom_routing_function=custom_routing_function,
            correction_bias=correction_bias,
            routed_scaling_factor=routed_scaling_factor,
        )

        # print(layer.w2_q_weight.shape)
        # print(layer.w2_q_weight[0,:,:])
        # print(layer.w2_q_weight[1,:,:])
        # print(layer.w2_q_scale_col[0,:,:])
        # print(layer.w2_q_scale_col[1,:,:])
        # exit()
         
        if x.shape[0] > 1:
            import torch.nn.functional as F
            hidden_dim = x.shape[1]

            final_hidden_states = torch.zeros(
                (x.shape[0], x.shape[1]), dtype=x.dtype, device=x.device
            )
            
    
            
            
            for topx in range(x.shape[0]):

                idx = 0  # idx 表示当前的 expect 在哪一个id
                for expert_idx in topk_ids[topx]:
                    
                    expert_layer_down = layer.w2_q_weight[expert_idx, :, :].squeeze()
                    expert_layer_gate_up = layer.w13_q_weight[expert_idx, :, :].squeeze()

                    scales_down = layer.w2_q_scale_col[expert_idx, :, :].squeeze().to(x.device, dtype=torch.float32)
                    scales_up = layer.w13_q_scale_col[expert_idx, :, :].squeeze().to(x.device, dtype=torch.float32)
                    
                    assert layer.hidden_size == hidden_dim, f"hidden_size {layer.hidden_size} != {hidden_dim}"
                    expert_layer_down = mixgemm.dequant(expert_layer_down, 
                    scales_down, hidden_dim, layer.intermediate_size_per_partition, 4, 128, 32, 4)

                    expert_layer_gate_up = mixgemm.dequant(expert_layer_gate_up, scales_up, 
                            layer.intermediate_size_per_partition * 2, hidden_dim, 
                            4, 128, 32, 4)  

                    current_state = x[topx, :].reshape(-1, hidden_dim)
                    from sglang.srt.layers.activation import SiluAndMul
                    act_fn = SiluAndMul()
                    tmp = torch.mm(current_state, expert_layer_gate_up.T) + torch.mm(current_state[:,layer.w13_ind[expert_idx, :, :].squeeze()], 
                                                                        layer.w13_weight_cache[expert_idx, :, :].squeeze().T)
                    
                    act = act_fn(tmp)

                    # print(act)
                    act = torch.mm(act, expert_layer_down.T)  + torch.mm(act[:,layer.w2_ind[expert_idx, :, :].squeeze()], 
                                                                        layer.w2_weight_cache[expert_idx, :, :].squeeze().T)
                    current_hidden_states = act * topk_weights[topx, idx, None]

                    # print(current_hidden_states)

                    final_hidden_states[topx:topx+1,:] += current_hidden_states.to(x.dtype).reshape(1, hidden_dim)

                    idx += 1
            return final_hidden_states
        
        if x.shape[0] == 1:

            
            import torch.nn.functional as F
            hidden_dim = x.shape[1]

            # final_hidden_states = torch.zeros(
            #     (x.shape[0], x.shape[1]), dtype=x.dtype, device=x.device
            # )
            # topx = 0

            # layer.w13_q_scale_col[:,:,:] = 1
            # topk_ids = topk_ids[0:1, 0: 1]
            # topk_ids[:,:] = 0

            # idx = 0  # idx 表示当前的 expect 在哪一个id
            # for expert_idx in topk_ids[topx]:

            #     expert_layer_down = layer.w2_q_weight[expert_idx, :, :].squeeze()
            #     expert_layer_gate_up = layer.w13_q_weight[expert_idx, :, :].squeeze()

            #     scales_down = layer.w2_q_scale_col[expert_idx, :, :].squeeze().to(x.device, dtype=torch.float32)
            #     scales_up = layer.w13_q_scale_col[expert_idx, :, :].squeeze().to(x.device, dtype=torch.float32)
                
            #     assert layer.hidden_size == hidden_dim, f"hidden_size {layer.hidden_size} != {hidden_dim}"
            #     expert_layer_down = mixgemm.dequant(expert_layer_down, 
            #     scales_down, hidden_dim, layer.intermediate_size_per_partition, 4, 128, 32, 4)

            #     expert_layer_gate_up = mixgemm.dequant(expert_layer_gate_up, scales_up, 
            #             layer.intermediate_size_per_partition * 2, hidden_dim, 
            #             4, 128, 32, 4)  

            #     current_state = x[topx, :].reshape(-1, hidden_dim)
            #     from sglang.srt.layers.activation import SiluAndMul
            #     act_fn = SiluAndMul()
            #     tmp = torch.mm(current_state, expert_layer_gate_up.T) + torch.mm(current_state[:,layer.w13_ind[expert_idx, :, :].squeeze()], 
            #                                                         layer.w13_weight_cache[expert_idx, :, :].squeeze().T)
            #     # tmp = torch.mm(current_state, expert_layer_gate_up.T)

            #     act = act_fn(tmp)

            #         # print(act)
            #     act = torch.mm(act, expert_layer_down.T)  + torch.mm(act[:,layer.w2_ind[expert_idx, :, :].squeeze()], 
            #                                                         layer.w2_weight_cache[expert_idx, :, :].squeeze().T)
            #     current_hidden_states = act * topk_weights[topx, idx, None]
                
            #     print("grand data")
            #     print(current_hidden_states)


            final_hidden_states_ = torch.zeros(
                (x.shape[0], x.shape[1]),
                    dtype=x.dtype, device=x.device)


            down_weight = layer.w2_q_weight
            gate_up = layer.w13_q_weight


            block_dim_x = 32
            block_dim_y = 4
            from sglang.srt.layers.activation import SiluAndMul
            act_fn = SiluAndMul()

            topx = 0
            out =   torch.zeros(
                (len(topk_ids[topx]),   gate_up.shape[1]), 
                dtype=x.dtype, 
                device=x.device)
            n_outliers = layer.fp_features_num
            group_size = 128

            current_state = x[topx, :].reshape(-1, hidden_dim)

            # mixgemm.gemv_int4_fp16_mix(1,  2 * layer.intermediate_size_per_partition, layer.hidden_size,
            #                            current_state, gate_up, out, block_dim_x, 
            #                            block_dim_y, layer.w13_q_scale_col.to(torch.float32), 
            #                            layer.w13_weight_cache, 
            #                            layer.w13_ind, n_outliers)

            moe_gemm.moe_gemv_i4(2 * layer.intermediate_size_per_partition, 
                        layer.hidden_size,
                        current_state,
                        out,
                        gate_up, 
                        down_weight,
                        topk_ids,
                        block_dim_x, 
                        block_dim_y,
                        layer.w13_q_scale_col,
                        layer.w13_weight_cache,
                        layer.w13_ind,
                        n_outliers,
                        group_size, 90)
            

            
            act = act_fn(out)

            # torch.save(act, "act.pt")
            # torch.save(down_weight.data, "down_weight.pt")
            # torch.save(layer.w2_q_scale_col.data, "w2_q_scale_col.pt")
            # torch.save(topk_weights.data, "topk_weights.pt")
            # torch.save(topk_ids, "topk_ids")
            # torch.save(layer.w2_weight_cache.data, "w2_weight_cache.pt")
            # torch.save(layer.w2_ind.data, "w2_ind.pt")
            # exit()

            moe_gemm.moe_gemv_down_i4(layer.hidden_size,
                                      layer.intermediate_size_per_partition,
                                      act,
                                      final_hidden_states_,
                                      down_weight,
                                      topk_weights,
                                        topk_ids,
                                        block_dim_x,
                                        block_dim_y,
                                        layer.w2_q_scale_col,
                                        layer.w2_weight_cache,
                                        layer.w2_ind,
                                        n_outliers,
                                        group_size)
            return final_hidden_states_
 
        
       



    @staticmethod
    def get_weight_loader(layer, weight_loader):



        def moe_mixq_weight_loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            weight_name: str,
            shard_id: str,
            expert_id: int,
        ):
            # print("moe_mixq_weight_loader----chenyidong")
            # print(weight_name)
            # print(shard_id)
            # print(expert_id)
            # print(param.data.shape)
            # print(loaded_weight.shape)



            shard_size = param.data.shape[1]

            if "ind" in weight_name:
                param.data[expert_id, :, : ] = loaded_weight
                return 
            
            if  "w13" in weight_name:
                if shard_id == "w1":
                    param.data[expert_id, : shard_size // 2] = loaded_weight
                else:
                    param.data[expert_id, shard_size // 2 :] = loaded_weight
                return 
            
            if "w2" in weight_name:
                param.data[expert_id, :, : ] = loaded_weight

                return 


            
            # weight_loader(param, 
            # loaded_weight, 
            # weight_name, 
            # shard_id, 
            # expert_id)


        return moe_mixq_weight_loader
