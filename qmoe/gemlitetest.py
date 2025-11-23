from gemlite.helper import *
device, dtype = 'cuda:0', torch.float16

import torch
in_features = 4096
out_features = 2048
group_size = 128
linear = torch.nn.Linear(in_features, out_features, bias=False, device=None, dtype=torch.float16).cuda()

from gemlite.helper import GemLiteLinearTriton, DType
from hqq.core.quantize import HQQLinear, BaseQuantizeConfig

orig_shape   = (out_features, in_features)
quant_config   = BaseQuantizeConfig(nbits=4, group_size=group_size)


hqq_layer    = HQQLinear(linear, quant_config=quant_config,
                          compute_dtype=torch.float16, device=device, 
                          del_orig=False) 


gemlite_linear = GemLiteLinearTriton(W_nbits=4, 
                                    group_size=group_size, in_features=in_features, out_features=out_features, 
                                    input_dtype=DType.FP16, output_dtype=DType.FP16)

gemlite_linear.pack(hqq_layer.unpack(dtype=torch.uint8).view(orig_shape),
                     hqq_layer.meta['scale'].clone(), 
                     hqq_layer.meta['zero'].clone(), bias=None)


