import bitblas
from bitblas.cache import global_operator_cache, get_database_path
from bitblas import auto_detect_nvidia_target
BITBLAS_TARGET = auto_detect_nvidia_target()
BITBLAS_DATABASE_PATH = get_database_path()
import torch
import torch.nn.functional as F
import torch.nn as nn

class BitLinear158(nn.Module):
    """
    This is only for training, and kernel optimization is needed for efficiency.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, input_bits: int = 8,
                 device=None, dtype=None, config=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_bits = input_bits
        matmul_config = bitblas.MatmulConfig(
            N=self.out_features,  # N dimension
            K=self.in_features,  # K dimension
            A_dtype="int8",  # activation A dtype
            W_dtype="int2",  # weight W dtype
            accum_dtype="int32",  # accumulation dtype
            out_dtype="float32",  # output dtype
            layout="nt",  # matrix layout, "nt" indicates the layout of A is non-transpose and the layout of W is transpose
            with_bias=False,  # bias
            # configs for weight only quantization
            group_size=None,  # setting for grouped quantization
            with_scaling=False,  # setting for scaling factor
            with_zeros=False,  # setting for zeros
            zeros_mode=None,  # setting for how to calculating zeros
        )
        ENABLE_TUNING = True
        self.bitblas_matmul = self._get_or_create_bitblas_operator(matmul_config, ENABLE_TUNING)
        self.Qp = 2**(self.input_bits - 1) - 1
        self.register_buffer(
            "weight",
            torch.randint(0, 2,
                (out_features, in_features),
                dtype=torch.int8,
                device=device,
            ),
        )
        self.register_buffer(
            "weight_scale",
            torch.randn(
                (1),
                dtype=torch.bfloat16,
                device=device,
            ),
        )
        if bias : 
            self.register_buffer(
                "bias",
                torch.zeros(
                    (self.out_features),
                    dtype=torch.bfloat16,
                    device=device
                ))
        else : 
            self.bias = None
        self.create_bitblas_weights()

    def _get_or_create_bitblas_operator(self, config, enable_tuning):
        if global_operator_cache.size() == 0:
            global_operator_cache.load_from_database(BITBLAS_DATABASE_PATH, BITBLAS_TARGET)

        bitblas_matmul = global_operator_cache.get(config)
        if bitblas_matmul is None:
            # should disable tuning for the first time because we may require loading bitblas operator from database.
            bitblas_matmul = bitblas.Matmul(config, target=BITBLAS_TARGET, enable_tuning=False)
            if enable_tuning:
                bitblas_matmul.hardware_aware_finetune(topk=20)
                global_operator_cache.add(config, bitblas_matmul)
                global_operator_cache.save_into_database(BITBLAS_DATABASE_PATH, BITBLAS_TARGET)
                print("BitBLAS Tuning done, appended operator to global_operator_cache.")
            else:
                print("BitBLAS Operator created.")
        else:
            print("BitBLAS Operator found in global_operator_cache.")
        return bitblas_matmul
    
    def create_bitblas_weights(self):
        qweight = self.bitblas_matmul.transform_weight(self.weight)
        qweight = nn.Parameter(qweight, requires_grad=False)
        self.register_buffer("qweight", qweight)
    
    @torch.compile
    def activation_quant(self, x, num_bits=8):
        Qn = -(2**(num_bits - 1))
        Qp = 2**(num_bits - 1) - 1
        s = Qp / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        result = (x * s).round().clamp(Qn, Qp)
        return result.type(torch.int8), s

    @torch.compile
    def post_quant_process(self, input, si, sw):
        out = input / si
        out = out / sw
        return out
    
    def simple_forward(self, x):
        w = self.weight
        w_quant = w.to(torch.bfloat16)
        x_quant, x_scale = self.activation_quant(x, self.input_bits)
        y = self.post_quant_process(F.linear(x_quant.to(torch.bfloat16), w_quant) , x_scale, self.weight_scale)
        if self.bias is not None : 
            y += self.bias.view(1, -1).expand_as(y)
        return y
    
    def forward(self, input):
        ref_result = self.simple_forward(input)
        print(ref_result)
        quant_input, si = self.activation_quant(input, self.input_bits)
        fp32_out = self.bitblas_matmul(quant_input, self.qweight)
        bf16_out = fp32_out.to(torch.bfloat16)
        sw = self.weight_scale
        out = self.post_quant_process(bf16_out, si, sw)
        print(out)
        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)
        torch.testing.assert_close(ref_result, out, rtol=1e-3, atol=1e-3)
        return out
    
size = 8192
linear = BitLinear158(size, size, False, 8, "hip")
x = torch.rand((8, size), dtype=torch.bfloat16).cuda()
linear(x)