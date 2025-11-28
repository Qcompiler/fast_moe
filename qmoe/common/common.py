import torch
import numpy as np
import torch.nn as nn

seed = 0
np.random.seed(seed)
torch.random.manual_seed(seed)


def generate_randint(k, out_dim, device, dtype = torch.float16):
    weight  = torch.randint(low=-8, high=8, size=(out_dim, k)).to(torch.float16).to(device)
    # weight  =  -torch.zeros((out_dim, k)).to(torch.float16).to(device) * 8
    # ind = 3
    # weight[:, ind:ind+1] = -8
    #   weight[out_dim-3,:] = 0
    #   weight[:, 0:32] = torch.randint(low=-8, high=8, size=(out_dim, 32)).to(torch.float16).to(device)
    #   vector =  torch.ones((1, k)).to(torch.float16).to(device) 
    # vector =  torch.zeros((1, k)).to(torch.float16).to(device) 
    # vector[0, ind:ind+1] = 1
    # vector[0, 0:8] = torch.randint(low=-3, high=3, size=(1, 8)).to(torch.float16).to(device)
  
    vector =  torch.randint(low=-2, high=2, size=(1, k)).to(torch.float16).to(device) / 100
    return weight, vector



def generate_randint_moe(num_experts, intermediate_size, k, top_k, device, dtype):
  
  
    # gate_up_weight  = torch.rand((num_experts,  2 * intermediate_size, k)).to(torch.float16).to(device) - 0.5

    # gate_up_weight  =  - ( torch.ones((num_experts, 2 * intermediate_size, k)).to(torch.float16).to(device) ) * 7

    gate_up_weight  =  torch.randint(low=-7, high=8, size= (num_experts, 2 * intermediate_size, k) ).to(dtype).to(device) 


    vector =  torch.randint(low=-1, high=2, size=(1, k)).to(dtype).to(device) / 10

    # vector =  torch.randint(low=-15, high=16, size=(1, k)).to(torch.float16).to(device) 
    # vector  = - ( torch.ones((1 , k)).to(torch.float16).to(device) )
    topk_ids = torch.randint(0, num_experts, (1, top_k) , device = device, dtype = torch.int64)  

    return gate_up_weight, vector, topk_ids


def generate_randint_moe_down(num_experts, intermediate_size, k, top_k, device, dtype):
  
  
    down_weight  = ( torch.rand((num_experts, k, intermediate_size)).to(dtype).to(device) - 0.5 ) / 10

    # down_weight  =  - ( torch.ones((num_experts, k, intermediate_size)).to(torch.float16).to(device) )  * 7


    topk_weights = (torch.rand(1, top_k, device = device, dtype = torch.float32) - 0.5) / 10

    # topk_weights =  torch.ones((1, top_k)).to(torch.float32).to(device)  

    topk_ids = torch.randint(0, num_experts, (1, top_k) , device = device, dtype = torch.int64)  

    return down_weight, topk_weights



def compute_moe_gate_up(x, topk_ids, gate_up):

    
    hidden_dim = x.shape[1]
    intermedia_dim = gate_up.shape[1]

    topx = 0
    final_hidden_states = torch.zeros(
        (len(topk_ids[topx]), intermedia_dim), dtype=x.dtype, device=x.device)

    
    for topx in range(x.shape[0]):

        idx = 0  # idx 表示当前的 expect 在哪一个id
        for expert_idx in topk_ids[topx]:
            
            expert_layer_gate_up = gate_up [expert_idx, :, :].squeeze()
            current_state = x[topx, :].reshape(-1, hidden_dim)

            tmp = torch.mm(current_state, expert_layer_gate_up.T)
     
            final_hidden_states[idx:idx+1,:] = tmp.reshape(1, intermedia_dim)

            idx += 1

    return final_hidden_states

import torch.nn.functional as F
class SiluAndMul(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

def compute_moe_gate_up_down(x, topk_ids, gate_up, down_weight, topk_weights):

    
    hidden_dim = x.shape[1]
    intermedia_dim = gate_up.shape[1]

    topx = 0
    final_hidden_states = torch.zeros(
        (x.shape[0], hidden_dim), dtype=x.dtype, device=x.device)


    for topx in range(x.shape[0]):

        idx = 0  # idx 表示当前的 expect 在哪一个id
        for expert_idx in topk_ids[topx]:
            
            expert_layer_gate_up = gate_up [expert_idx, :, :].squeeze()
            expert_layer_down = down_weight[expert_idx, :, :].squeeze()
            current_state = x[topx, :].reshape(-1, hidden_dim)

            tmp = torch.mm(current_state, expert_layer_gate_up.T)
            act_fn = SiluAndMul()

            # print(tmp.shape)
            act = act_fn(tmp)
            # print(act.shape)
            # print(expert_layer_down)
            act = torch.mm(act, expert_layer_down.T)

            # print(act.shape)
            # exit()
            
                    
            current_hidden_states = act * topk_weights[topx, idx, None]

            final_hidden_states[topx:topx+1,:] += current_hidden_states.to(x.dtype).reshape(1, hidden_dim)

            # print(final_hidden_states.shape)
            idx += 1


    return final_hidden_states

def compute_moe_gate_up_down_opt(x, topk_ids, gate_up,  down_weight, topk_weights):

    import moe_gemm
    hidden_dim = x.shape[1]


    final_hidden_states_ = torch.zeros(
        (x.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)


    block_dim_x = 32
    block_dim_y = 4

    topx = 0
    current_state = x[topx, :].reshape(-1, hidden_dim)
    out =   torch.zeros(
        ( len(topk_ids[topx]), gate_up.shape[1]), dtype=x.dtype, device=x.device)
    moe_gemm.moe_gemv(current_state,
                  out,
                gate_up, 
                gate_up,
                topk_ids,
                block_dim_x, 
                block_dim_y )
    act_fn = SiluAndMul()
    act = act_fn(out)

    moe_gemm.moe_gemv_down(act,
                final_hidden_states_,
                down_weight,
                topk_weights,
                topk_ids,
                block_dim_x, 
                block_dim_y )
    return final_hidden_states_

def compute_moe_gate_up_opt(x, topk_ids, gate_up):

    import moe_gemm
    hidden_dim = x.shape[1]

    out = torch.zeros(
        (1, x.shape[1]), dtype=x.dtype, device=x.device)


    block_dim_x = 32
    block_dim_y = 4

    topx = 0
    current_state = x[topx, :].reshape(-1, hidden_dim)
    out =   torch.zeros(
        ( len(topk_ids[topx]), gate_up.shape[1]), dtype=x.dtype, device=x.device)
    moe_gemm.moe_gemv(current_state,
                  out,
                gate_up, 
                gate_up,
                topk_ids,
                block_dim_x, 
                block_dim_y )

    return out



def _get_perms():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])

    perm = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return perm, scale_perm, scale_perm_single

_perm, _scale_perm, _scale_perm_single = _get_perms()
class Layer(nn.Module):
    """PyTorch compatible Marlin layer; 4-bit (symmetric grouped) linear layer without bias."""

    def __init__(self, infeatures, outfeatures, groupsize=-1, tile = 16):
        """Create an empty Marlin layer.
        @infeatures: number of input features (must be divisible by 128)
        @outfeatures: number of output features (must be divisible by 256)
        @groupsize: quantization groupsize (must be -1 or 128)
        """
        super().__init__()
        if groupsize not in [-1, 128]:
            raise ValueError('Only groupsize -1 and 128 are supported.')
        if infeatures % 128 != 0 or outfeatures % 256 != 0:
            raise ValueError('`infeatures` must be divisible by 128 and `outfeatures` by 256.')
        if groupsize == -1:
            groupsize = infeatures
        if infeatures % groupsize != 0:
            raise ValueError('`infeatures` must be divisible by `groupsize`.')
        self.k = infeatures
        self.n = outfeatures
        self.groupsize = groupsize
        self.register_buffer('B', torch.empty((self.k // tile, self.n * tile // 8), dtype=torch.int))
        self.register_buffer('s', torch.empty((self.k // groupsize, self.n), dtype=torch.half))
        # 128 is currently the minimum `tile_n`, hence it gives the maximum workspace size; 16 is the default `max_par`
        self.register_buffer('workspace', torch.zeros(self.n // 128 * 16, dtype=torch.int), persistent=False)

    def forward(self, A):
        C = torch.empty(A.shape[:-1] + (self.s.shape[1],), dtype=A.dtype, device=A.device)
        mul(A.view((-1, A.shape[-1])), self.B, C.view((-1, C.shape[-1])), self.s, self.workspace)
        return C

    def pack(self, linear, scales, tile):
        """Pack a fake-quantized linear layer into this actual Marlin representation.
        @linear: fake-quantized `torch.nn.Linear` layer to convert (must be of type `torch.half`)
        @scales: corresponding quantization scales of shape `(infeatures, groups)`
        """ 
        if linear.weight.dtype != torch.half:
            raise ValueError('Only `torch.half` weights are supported.')
        maxq = 2 ** 4 - 1
        s = scales.t()
        w = linear.weight.data.t()
        #print(w.shape)
        if self.groupsize != self.k:
            w = w.reshape((-1, self.groupsize, self.n))
            w = w.permute(1, 0, 2)
            w = w.reshape((self.groupsize, -1))
            s = s.reshape((1, -1))
        w = torch.round(w / s).int()
        w += (maxq + 1) // 2
        w = torch.clamp(w, 0, maxq)
        if self.groupsize != self.k:
            w = w.reshape((self.groupsize, -1, self.n))
            w = w.permute(1, 0, 2)
            w = w.reshape((self.k, self.n)).contiguous()
            s = s.reshape((-1, len(_scale_perm)))[:, _scale_perm]
        else:
            s = s.reshape((-1, len(_scale_perm_single)))[:, _scale_perm_single]
        s = s.reshape((-1, self.n)).contiguous()
        w = w.reshape((self.k // tile, tile, self.n // tile, tile))
        w = w.permute((0, 2, 1, 3))
        w = w.reshape((self.k // tile, self.n * tile))
        res = w
        res = res.reshape((-1, _perm.numel()))[:, _perm].reshape(res.shape)
        q = np.zeros((res.shape[0], res.shape[1] // 8), dtype=np.uint32)
        
        # print(w.shape)
        res = res.cpu().numpy().astype(np.uint32)
        # print(res.shape)

        for i in range(8):
            q |= res[:, i::8] << 4 * i
        q = torch.from_numpy(q.astype(np.int32)).to(w.device)
        # print(q.shape)
        # exit()
        # print("target shape")
        # print(self.B.shape)
        # exit()
        self.B[:, :] = q.to(self.B.device)
        self.s[:, :] = s.to(self.s.device)

def gen_quant4(m, n, w, groupsize=-1):
    DEV = w.device
    tile = 16
    maxq = 2 ** 4
    if groupsize != -1:
        w = w.reshape((-1, groupsize, n))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))
    s = torch.max(torch.abs(w), 0, keepdim=True)[0]
    s *= 2 / maxq
    w = torch.round(w / s).int()


    w += (maxq + 1) // 2
    w = torch.clamp(w, 0, maxq - 1)
    ref = (w - (maxq + 1) // 2).half() * s
    if groupsize != -1:
        def reshape(w):
            w = w.reshape((groupsize, -1, n))
            w = w.permute(1, 0, 2)
            w = w.reshape((m, n)).contiguous()
            return w
        ref = reshape(ref)
        w = reshape(w)
    s = s.reshape((-1, n)).contiguous()
    linear = nn.Linear(m, n)
    linear.weight.data = ref.t()
    # Workaround to test some special cases that are forbidden by the API
    layer = Layer(256, 256, groupsize=groupsize)
    if groupsize == -1:
        groupsize = m
    layer.k = m
    layer.n = n
    layer.groupsize = groupsize
    layer.B = torch.empty((m // tile, n * tile // 8), dtype=torch.int, device=DEV)
    layer.s = torch.empty((m // groupsize, n), dtype=torch.half, device=DEV)
    layer.pack(linear, s.t(), tile)
    q = layer.B
    s = layer.s
    return ref, q, s
"export PYTHONPATH=/home/chenyidong/newstart/bandwidth/jitcu"

class MyLayer(nn.Module):
    """PyTorch compatible Marlin layer; 4-bit (symmetric grouped) linear layer without bias."""

    def __init__(self, infeatures, outfeatures, groupsize=-1, tile = 1):
        """Create an empty Marlin layer.
        @infeatures: number of input features (must be divisible by 128)
        @outfeatures: number of output features (must be divisible by 256)
        @groupsize: quantization groupsize (must be -1 or 128)
        """
        super().__init__()
        # if groupsize not in [-1, 128, outfeatures]:
        #     raise ValueError('Only groupsize -1 and 128 are supported.')
        # if infeatures % 128 != 0 or outfeatures % 256 != 0:
        #     raise ValueError('`infeatures` must be divisible by 128 and `outfeatures` by 256.')
        if groupsize == -1:
            groupsize = infeatures
        if infeatures % groupsize != 0:
            raise ValueError('`infeatures` must be divisible by `groupsize`.')
        self.k = infeatures
        self.n = outfeatures
        self.groupsize = groupsize
        self.register_buffer('B', torch.empty((self.n * tile // 8 , self.k // tile), dtype=torch.int))
        self.register_buffer('s', torch.empty((self.k // groupsize, self.n), dtype=torch.half))

    def pack(self, linear, scales, tile):

        
        
        k = self.k
        
        interleave = []
        for i in range(k//8):
            out = [0, 2, 4, 6, 1, 3, 5, 7]
            for j in range(8):
                out[j] = out[j] + 8 * i
            interleave += out
        interleave = np.array(interleave)
        w = linear
        
        res = w[:,interleave]
      

      
        q = np.zeros((res.shape[0], res.shape[1] // 8), dtype=np.uint32)
   
        res = res.cpu().numpy().astype(np.uint32)

        for i in range(8):
            q |= res[:, i::8] << 4 * i
        q = torch.from_numpy(q.astype(np.int32)).to(w.device)

        self.B[:, :] = q.to(self.B.device)
        # self.s[:, :] = s.to(self.s.device)
    def pack_without_reorder(self, linear, scales, tile):

        
        
        k = self.k
        
        interleave = []
        for i in range(k//8):
            out = [0, 1, 2, 3, 4, 5, 6, 7]
            for j in range(8):
                out[j] = out[j] + 8 * i
            interleave += out
        interleave = np.array(interleave)
        w = linear
        
        res = w[:,interleave]
      

      
        q = np.zeros((res.shape[0], res.shape[1] // 8), dtype=np.uint32)
   
        res = res.cpu().numpy().astype(np.uint32)

        for i in range(8):
            q |= res[:, i::8] << 4 * i
        q = torch.from_numpy(q.astype(np.int32)).to(w.device)

        self.B[:, :] = q.to(self.B.device)
def gen_quant4_my(n, k, w, groupsize=-1,  tile = 1):
    if groupsize == -1:
        groupsize = k
    DEV = w.device   
    maxq = 2 ** 4   # 4-bit量化，最大值15
    n, k = w.shape  # 原始权重矩阵形状

    # 计算需要的组数（向上取整）
    num_groups = (k + (groupsize-1)) // groupsize  # 等价于 math.ceil(k / 128)

    # 填充权重矩阵，使k能被128整除
    padded_k = num_groups * groupsize
    if k % groupsize != 0:
        w_padded = torch.nn.functional.pad(w, (0, padded_k - k))
    else:
        w_padded = w

    # 将权重矩阵重塑为 (n, num_groups, 128)
    w_reshaped = w_padded.reshape(n, num_groups, groupsize)
    # 计算每个组的缩放因子s (n, num_groups, 1)
    s = torch.max(torch.abs(w_reshaped), dim=2, keepdim=True)[0]
    s *= 2 / maxq  # 缩放因子范围调整
    # 量化过程
    linear = torch.clone(w_reshaped)
    linear = torch.round(linear / s).int()

    linear += (maxq + 1) // 2  # 添加零点偏移


    linear = torch.clamp(linear, 0, maxq - 1)
    # 将量化的权重和缩放因子重塑回原始形状
    linear = linear.reshape(n, -1)[:, :k]  # 移除填充并恢复原始k维度
    s = s.reshape(n, -1).contiguous()  # 缩放因子形状为 (n, num_groups)  

    # Workaround to test some special cases that are forbidden by the API
    layer = MyLayer(k, n, groupsize=groupsize, tile = tile)

    layer.k = k
    layer.n = n
    layer.groupsize = groupsize
    layer.B = torch.empty((n // tile , k  * tile // 8), dtype=torch.int, device=DEV)
    layer.pack(linear, s.t(), tile = tile)
    q = layer.B



    return q, s

def gen_quant4_my_no_reorder(n, k, w, groupsize=-1,  tile = 1):
    if groupsize == -1:
        groupsize = k
    DEV = w.device   
    maxq = 2 ** 4   # 4-bit量化，最大值15
    n, k = w.shape  # 原始权重矩阵形状

    # 计算需要的组数（向上取整）
    num_groups = (k + (groupsize-1)) // groupsize  # 等价于 math.ceil(k / 128)

    # 填充权重矩阵，使k能被128整除
    padded_k = num_groups * groupsize
    if k % groupsize != 0:
        w_padded = torch.nn.functional.pad(w, (0, padded_k - k))
    else:
        w_padded = w

    # 将权重矩阵重塑为 (n, num_groups, 128)
    w_reshaped = w_padded.reshape(n, num_groups, groupsize)
    # 计算每个组的缩放因子s (n, num_groups, 1)
    s = torch.max(torch.abs(w_reshaped), dim=2, keepdim=True)[0]
    s *= 2 / maxq  # 缩放因子范围调整
    # 量化过程
    linear = torch.clone(w_reshaped)
    linear = torch.round(linear / s).int()

    linear += (maxq + 1) // 2  # 添加零点偏移


    linear = torch.clamp(linear, 0, maxq - 1)
    # 将量化的权重和缩放因子重塑回原始形状
    linear = linear.reshape(n, -1)[:, :k]  # 移除填充并恢复原始k维度
    s = s.reshape(n, -1).contiguous()  # 缩放因子形状为 (n, num_groups)  

    # Workaround to test some special cases that are forbidden by the API
    layer = MyLayer(k, n, groupsize=groupsize, tile = tile)

    layer.k = k
    layer.n = n
    layer.groupsize = groupsize
    layer.B = torch.empty((n // tile , k  * tile // 8), dtype=torch.int, device=DEV)
    layer.pack_without_reorder(linear, s.t(), tile = tile)
    q = layer.B



    return q, s

def compute_moe(x, topk_ids, down_weight, gate_up, topk_weights):

    
    hidden_dim = x.shape[1]

    final_hidden_states = torch.zeros(
        (x.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)


    for topx in range(x.shape[0]):

        idx = 0  # idx 表示当前的 expect 在哪一个id
        for expert_idx in topk_ids[topx]:
            
            expert_layer_down = down_weight[expert_idx, :, :].squeeze()
            expert_layer_gate_up = gate_up [expert_idx, :, :].squeeze()
            current_state = x[topx, :].reshape(-1, hidden_dim)
            act_fn = SiluAndMul()
            tmp = torch.mm(current_state, expert_layer_gate_up.T)
            act = act_fn(tmp)
            act = torch.mm(act, expert_layer_down.T)
            current_hidden_states = act * topk_weights[topx, idx, None]

            final_hidden_states[topx:topx+1,:] += current_hidden_states.to(x.dtype).reshape(1, hidden_dim)

            idx += 1

    return final_hidden_states


def import_code(filename):
  file = open(filename)
  code = file.read()
  return code