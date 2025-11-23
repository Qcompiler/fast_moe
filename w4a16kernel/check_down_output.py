
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
    act_fn = SiluAndMul()
    topx = 0
    current_state = x[topx, :].reshape(-1, hidden_dim)
    final_hidden_states_ = torch.zeros(
        (x.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
    out =   torch.zeros(
        ( len(topk_ids[topx]), gate_up.shape[1]), dtype=x.dtype, device=x.device)
    moe_gemm.moe_gemv(current_state,
                  out,
                gate_up, 
                down_weight,
                topk_ids,
                block_dim_x, 
                block_dim_y )
    act = act_fn(out)

    print(act)
    print(act.shape)
    print(final_hidden_states.shape)
    print(down_weight.shape)
    print(topk_ids)
    print(topk_ids.shape)
    
    moe_gemm.moe_gemv_down(act,
                  final_hidden_states_,
                down_weight,
                topk_weights,
                topk_ids,
                block_dim_x, 
                block_dim_y )
    print(final_hidden_states_)
    
    
    print("----debug")

    import  moe_gemm
    block_dim_x = 32
    block_dim_y = 4
    # act = torch.load("../act.pt","cuda")
    final_hidden_states_ *= 0
    # final_hidden_states_ = torch.load("../final_hidden_states_.pt","cuda")

    topk_weights_ = torch.load("../topk_weights.pt","cuda")

    # down_weight_ = torch.load("../down_weight.pt","cuda")

    print("topk weight")
    print(topk_weights_)
    print(topk_weights_.dtype)
    print("my topk weight")
    print(topk_weights)

    moe_gemm.moe_gemv_down(act,
                            final_hidden_states_,
                            down_weight,
                            topk_weights_,
                            topk_ids,
                            block_dim_x, 
                            block_dim_y )   
    print(final_hidden_states_)
    exit()

    return final_hidden_states_




# Define input dimensions 0
batch_size = 1
hidden_dim = 2048
intermediate_size = 1408
num_experts = 64
top_k = 1  # number of experts to route to per token

compute_dtype = torch.bfloat16
device = "cuda"
# x = torch.rand(batch_size, hidden_dim, device = device, dtype = compute_dtype)  
skew_factor = 0.2
x = torch.empty(batch_size, hidden_dim, device = device, dtype = compute_dtype).exponential_(skew_factor)
x = x - torch.mean(x)

topk_ids = torch.randint(0, num_experts, (batch_size, top_k) , device = device, dtype = torch.int32)  

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


topk_weights = torch.rand(batch_size, top_k, device = device, dtype = torch.float32)   / 100


output = compute_moe(x, topk_ids, down, gate_up)



output2 = compute_moe_opt(x, topk_ids, down, gate_up)

ind = ( (output - output2).abs().argmax())

print("check output")
print(output)
print(output2)
print("check done\n\n")

print( (output - output2).abs())

print( (output - output2).abs().max() )


assert( (output - output2).abs().max() <= 0.1)

print(output[ int(ind / len(output[0])), ind %  len(output[0])])
print(output2[ int(ind / len(output[0])), ind %  len(output[0])])