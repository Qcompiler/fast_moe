import argparse
parser = argparse.ArgumentParser(description='Calculate volume of a cylinder')
# 添加参数
parser.add_argument('--down', action='store_true')

parser.add_argument('--gate', action='store_true')

args = parser.parse_args()

if args.down == 1:
    file = "moe_gemv_down.cu"

if args.gate == 1:

    file = "moe_gemv_gate.cu"
f = open(file)
data = f.readlines()


output = open("generated/i4_" + file.replace("_template",""), 'w+')

print(data)

for i in range(len(data)):


    code = "half2 reg_x_0 = HALF2( shmem_vector  [k + 0]);"
    if  code in data[i]:
        data[i] = data[i].replace(code, "half2 reg_x_0[4];*(((int4 *)reg_x_0)) =  *(((int4 *)shmem_vector) +  k);")
    
    code = "half2 reg_x_0 = HALF2(( shmem_vector + (warp_id % each_warp_reduce_compute) * K)  [k + 0]);"
    if  code in data[i]:
        data[i] = data[i].replace(code, "half2 reg_x_0[4];*(((int4 *)reg_x_0)) =  *(((int4 *)( shmem_vector +  (warp_id % each_warp_reduce_compute) * K)) +  k);")


    code = "half2 reg_x_1 = HALF2( shmem_vector  [k + 2]);"
    if code in data[i]:
        data[i] = data[i].replace(code, "half2 reg_x_1[4];*(((int4 *)reg_x_1)) =  *(((int4 *)shmem_vector) +  k + 1);")

    code = "half2 reg_x_1 = HALF2(( shmem_vector + (warp_id % each_warp_reduce_compute) * K)  [k + 2]);"
    if code in data[i]:
        data[i] = data[i].replace(code, "half2 reg_x_1[4];*(((int4 *)reg_x_1)) =  *(((int4 *)(shmem_vector + (warp_id % each_warp_reduce_compute) * K)) +  k + 1);")


    code = "auto reg = "
    code_ = "uint2 reg = "
    if code in data[i]:
        data[i] = data[i].replace(code, code_)

    code = "half2 reg_a_0 = *(reinterpret_cast<half2 *>(&reg)  );"
    code_ = "half2 reg_a_0[4]; int reg_a_0_int = *(reinterpret_cast<int *>(&reg)  ); dequant(reg_a_0_int, reg_a_0); "
    if code in data[i]:
        data[i] = data[i].replace(code, code_)   

    code = "half2 reg_a_1 = *(reinterpret_cast<half2 *>(&reg) + 1 );"
    code_ = "half2 reg_a_1[4]; int reg_a_1_int = *(reinterpret_cast<int *>(&reg ) + 1 ); dequant(reg_a_1_int, reg_a_1); "
    if code in data[i]:
        data[i] = data[i].replace(code, code_)  

    code = "sum[topx] += "

    if code in data[i]:
        data[i] = "\t\t\t\t\t\t\t\t for (int kk = 0; kk < 4; ++kk) \n" + data[i]

        data[i] = data[i].replace("sum[topx]", "tmp")     
        data[i] = data[i].replace("reg_x_0", "reg_x_0[kk]")      
        data[i] = data[i].replace("reg_a_0", "reg_a_0[kk]")   
        data[i] = data[i].replace("reg_x_1", "reg_x_1[kk]")      
        data[i] = data[i].replace("reg_a_1", "reg_a_1[kk]")   



        data[i] += "\t\t\t\t\t\t\t\t\t tmp *=   scales_ptr[  (k * 8 ) / 128];   sum[topx] +=  tmp;"   

    code = "for (int w = 0; w < NUM_WARPS; ++w) {"
    if code in data[i]:
        data[i] = data[i] +  "\n \t\t\t float tmp = 0.0;"



        code_ = "\n \t\t\t float* scales_ptr =  (float *) (  (int4 *) scales + target_expect * ( ( M * K * 8) / ( 128 * 4 ) ) +  m * ( (K  * 8) / ( 128 * 4 )) ) ;"  

        data[i] += code_
    
    code = "int total_iterations"
    if code in data[i]:
        data[i] += "\n \t\t\t  total_iterations *=  ((sizeof(int32_t) / sizeof(int8_t)) * 2);"

    code = "int M, int K"
    code_ = "float* scales, int group_size,  int M, int K"
    if code in data[i]:
        data[i] = data[i].replace(code, code_)   

    code = "M, K"
    code_ = "scales, group_size, M, K"
    if code in data[i]:
        data[i] = data[i].replace(code, code_) 

    code = "output_dim, hidden_size" 
    code_ = "scales.data_ptr<float>(), group_size, output_dim, hidden_size"
    if code in data[i]:
        data[i] = data[i].replace(code, code_) 

    code = "int kernel_type" 
    code2 = "const jc::Tensor&"
    code_ = "const jc::Tensor& scales, int group_size, int kernel_type"
    if code in data[i] and code2 in data[i]:
        data[i] = data[i].replace(code, code_) 


    code = "down.data_ptr<half>()"
    if code in data[i]:
        data[i] = data[i].replace(code, "down.data_ptr<int32_t>()") 
    code = "gate_up.data_ptr<half>()"
    if code in data[i]:
        data[i] = data[i].replace(code, "gate_up.data_ptr<int32_t>()") 

    code = "const half* down, "
    if code in data[i]:
        data[i] = data[i].replace(code, "const int32_t* down, ") 
        
    code = "const half* __restrict__ a"
    if code in data[i]:
        data[i] = data[i].replace(code, "const int32_t* __restrict__ a") 

    code = "const half * target_mat"
    if code in data[i]:
        data[i] = data[i].replace(code, "const int32_t * target_mat") 
    
    code = "const half* d_A"
    if code in data[i]:
        data[i] = data[i].replace(code, "const int32_t* d_A") 
        

    code = "int k = (w * WARP_SIZE + lane) * 4;"
    code_ = "int k = (w * WARP_SIZE + lane) * NUM_PER_THREAD;"
    if code in data[i]:
        data[i] = data[i].replace(code, code_) 
    
    code = "int NUM_WARPS = (((K + WARP_SIZE - 1) / WARP_SIZE) + 4 - 1) / 4;"
    code_ = "const int NUM_PER_THREAD = 2; int NUM_WARPS = (((K + WARP_SIZE - 1) / WARP_SIZE) + NUM_PER_THREAD - 1) / NUM_PER_THREAD;"
    if code in data[i]:
        data[i] = data[i].replace(code, code_)     

    code = "int sharedMemSize"
    if code in data[i]:
        data[i] = data[i] + "\n sharedMemSize = sharedMemSize * 8;"
        data[i] = data[i] + """\n assert(group_size == 128);"""
    
    code = "*(((int4 *)shmem_vector) + i)"
    if code in data[i]:
        data[i] = data[i].replace("K", "(K * 8)")


    code = "(warp_id % each_warp_reduce_compute) * K"
    if  code in data[i]:
        data[i] = data[i].replace(code, "(warp_id % each_warp_reduce_compute) * (K * 8)")
        
out = output.write("".join(data))