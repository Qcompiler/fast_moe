import argparse
parser = argparse.ArgumentParser(description='Calculate volume of a cylinder')
# 添加参数
parser.add_argument('--down', action='store_true')

parser.add_argument('--gate', action='store_true')
parser.add_argument('--quant', action='store_true')

args = parser.parse_args()

if args.down == 1:
    file = "moe_gemv_down.cu"

if args.gate == 1:

    file = "moe_gemv_gate.cu"

output = "generated/bf16_" + file
if args.quant == 1:

    output = "generated/i4_bf16_" + file
    file = "generated/i4_" + file
    
f = open(file)
data = f.readlines()


output = open(output, 'w+')

print(data)

for i in range(len(data)):

    code = "#include <cuda_fp16.h> "
    if code in data[i]:
        data[i] += "#include <cuda_bf16.h>\n" 

    code = "<<<"
    if code in data[i]:
        data[i] = data[i].replace("d_A", "d_A_")
        data[i] = data[i].replace("d_B", "d_B_")
        data[i] = data[i].replace("d_C", "d_C_")

    code = "dim3 grid;"
    if code in data[i]:
        if not args.quant:
            data[i] += "\n \t \t const __nv_bfloat16 * d_A_ = reinterpret_cast<const __nv_bfloat16*>(d_A); \n"
        else:
            data[i] += "\n \t \t auto * d_A_ = d_A; \n" 
        data[i] += "\n \t \t const __nv_bfloat16 * d_B_ = reinterpret_cast<const __nv_bfloat16*>(d_B); \n" 
        data[i] += "\n \t \t  __nv_bfloat16 * d_C_ = reinterpret_cast< __nv_bfloat16*>(d_C); \n" 


    code = "half* __restrict__"
    code_ =  "__nv_bfloat16* __restrict__"
    if code in data[i]:
        data[i] = data[i].replace(code, code_)

    code = "__float2half"
    code_ =  "__float2bfloat16"
    if code in data[i]:
        data[i] = data[i].replace(code, code_)


    code = "half shmem_vector"
    code_ =  "__nv_bfloat16 shmem_vector"
    if code in data[i]:
        data[i] = data[i].replace(code, code_)

    code = "const half *"
    code_ =  "const __nv_bfloat16 *"
    if code in data[i]:
        data[i] = data[i].replace(code, code_)

    code = "HALF2"
    code_ =  "BFLOAT2"
    if code in data[i]:
        data[i] = data[i].replace(code, code_)

    if args.quant:
        code = "half2"
        code_ =  "__nv_bfloat162"
        if code in data[i] and "reg_a_" not in data[i]:
            data[i] = data[i].replace(code, code_)
    else:
        code = "half2"
        code_ =  "__nv_bfloat162"
        if code in data[i]:
            data[i] = data[i].replace(code, code_)        
    
 
out = output.write("".join(data))