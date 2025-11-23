
import torch.nn.functional as F
import torch
from torch import nn

torch.manual_seed(42)



class SiluAndMul(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]
def compute_moe(x, topk_ids, down_weight, gate_up):

    
    hidden_dim = x.shape[1]
    intermedia_dim = gate_up.shape[1]

    topx = 0
    final_hidden_states = torch.zeros(
        (len(topk_ids[topx]), intermedia_dim), dtype=x.dtype, device=x.device)

    
    for topx in range(x.shape[0]):

        idx = 0  # idx 表示当前的 expect 在哪一个id
        for expert_idx in topk_ids[topx]:
            
            expert_layer_down = down_weight[expert_idx, :, :].squeeze()
            expert_layer_gate_up = gate_up [expert_idx, :, :].squeeze()
            current_state = x[topx, :].reshape(-1, hidden_dim)

            tmp = torch.mm(current_state, expert_layer_gate_up.T)
     
            final_hidden_states[idx:idx+1,:] = tmp.reshape(1, intermedia_dim)

            idx += 1

    return final_hidden_states



def compute_moe_opt(x, topk_ids, down_weight, gate_up):

    
    hidden_dim = x.shape[1]

    final_hidden_states = torch.zeros(
        (x.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)

    out = torch.zeros(
        (1, x.shape[1]), dtype=x.dtype, device=x.device)

    tmp = torch.zeros(
        (1, gate_up.shape[1]), dtype=x.dtype, device=x.device)
    import moe_gemm

    block_dim_x = 32
    block_dim_y = 4

    topx = 0
    current_state = x[topx, :].reshape(-1, hidden_dim)
    out =   torch.zeros(
        ( len(topk_ids[topx]), gate_up.shape[1]), dtype=x.dtype, device=x.device)
    moe_gemm.moe_gemv(current_state,
                  out,
                gate_up, 
                down_weight,
                topk_ids,
                block_dim_x, 
                block_dim_y )

    return out

# Define input dimensions 0
batch_size = 1
hidden_dim = 2048
intermediate_size = 1024
num_experts = 32
top_k = 6  # number of experts to route to per token

compute_dtype = torch.float16
device = "cuda"
# x = torch.rand(batch_size, hidden_dim, device = device, dtype = compute_dtype)  
skew_factor = 0.1
x = torch.empty(batch_size, hidden_dim, device = device, dtype = compute_dtype).exponential_(skew_factor)
x = x - torch.mean(x)

topk_ids = torch.randint(0, num_experts, (batch_size, top_k) , device = device, dtype = torch.int64)  

# gate_up = torch.rand(num_experts,  2 * intermediate_size, hidden_dim , device = device, dtype = compute_dtype) / 100 
gate_up =  torch.empty(num_experts,  2 * intermediate_size, hidden_dim , 
                       device = device, 
                       dtype = compute_dtype).exponential_(skew_factor) / 100 
gate_up = gate_up - torch.mean(gate_up)


down = torch.rand(num_experts, 
            hidden_dim, intermediate_size , 
            device = device, 
            dtype = compute_dtype)  / 100

# down =  torch.empty(num_experts, hidden_dim, intermediate_size , device = device, dtype = compute_dtype).exponential_(skew_factor)


topk_weights = torch.rand(batch_size, top_k, device = device, dtype = compute_dtype)   / 100


output = compute_moe(x, topk_ids, down, gate_up)


output2 = compute_moe_opt(x, topk_ids, down, gate_up)



ind = ( (output - output2).abs().argmax())

print( (output - output2).abs().max() )


assert( (output - output2).abs().max() <= 0.1)

print(output[ int(ind / len(output[0])), ind %  len(output[0])])
print(output2[ int(ind / len(output[0])), ind %  len(output[0])])

