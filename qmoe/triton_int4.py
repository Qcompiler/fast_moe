



import torch, math, random, copy
from torch import Tensor
import triton
import triton.language as tl
@triton.autotune(
    configs=get_autotune_config(),
    key = KEYS,
    restore_value = ['a_ptr', 'b_ptr', 'c_ptr'],
    prune_configs_by = {'early_config_prune': kernel_config_pruner},
    use_cuda_graph = AUTOTUNE.USE_CUDA_GRAPH,
)


@triton.jit
def gemv_INT_kernel(
    a_ptr, b_ptr, c_ptr,
    scales_ptr, zeros_ptr, scales_a_ptr,
    mapping_ptr,
    M, N, K, 
    ######### Quant parms #########
    W_nbits: tl.constexpr, 
    group_size: tl.constexpr, 
    unpack_mask: tl.constexpr, 
    elements_per_sample: tl.constexpr, 
    type_id: tl.constexpr,
    use_prehook: tl.constexpr,
    ######### Strides #########
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_meta_a_m, stride_meta_a_g,
    stride_meta_g, stride_meta_n,
    ######### Dtypes #########
    input_dtype: tl.constexpr,
    output_dtype: tl.constexpr,
    acc_dtype: tl.constexpr,
    meta_dtype: tl.constexpr,
    ######### Meta-data mode #########
    channel_scale_mode: tl.constexpr,
    W_group_mode: tl.constexpr,
    zero_is_scalar: tl.constexpr,
    ######### tuning params #########
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    A_load_order: tl.constexpr, NUM_STAGES: tl.constexpr,
    dot_prod_mode:tl.constexpr,
    data_contiguous: tl.constexpr,
    dump_b_val: tl.constexpr = 0, #Improve accuracy mainly for A16W8 with post looop scaling
    #####################################
    meta_evict_policy: tl.constexpr = '',
    atomic_mode: tl.constexpr = 'relaxed',
    a_evict: tl.constexpr = 'evict_last',
    b_evict: tl.constexpr = 'evict_first',
    join_version: tl.constexpr = False,
    #################################
    load_scales_as_block: tl.constexpr = False,
):
    """
    GEMV for C = matmul(A, dequantize(B, scales, zeros)). This is optimized for M==1
    A is of shape (M, K): float16 or bfloat16
    B is of shape (K // elements_per_sample, N): int32 as a packed matrix
    C is of shape (M, N): float16 or bfloat16 depending on the input A
    scales and zeros is of shape (group_size, N): float16 or bfloat16
    """    

    pid   = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1) 
    pid_m, pid_n = pid % M, pid // M

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) 
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    #Vectorized coalesced load
    ##############################
    if data_contiguous:
        offs_bn = offs_n  
    else:
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N) 
    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_ak = offs_k
    offs_bk = offs_k
    ###############################

    a_ptrs  = a_ptr + offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak 

    if(join_version):
        BLOCK_SIZE_K_E: tl.constexpr = BLOCK_SIZE_K // elements_per_sample
        offs_bk = pid_k * BLOCK_SIZE_K_E + tl.arange(0, BLOCK_SIZE_K_E) 
        b_ptrs  = b_ptr + offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn 
    else:
        #orig version
        b_ptrs  = b_ptr + (offs_bk[:, None] // elements_per_sample) * stride_bk + offs_bn[None, :] * stride_bn 

    ###################################################################
    #Load
    if(A_load_order == 0):
        a = tl.load(a_ptrs, eviction_policy=a_evict)

    b = tl.load(b_ptrs, eviction_policy=b_evict)

    if(A_load_order == 1):
        a = tl.load(a_ptrs, eviction_policy=a_evict)
    
    if(W_group_mode > 0):
        k_m = (pid_k * (BLOCK_SIZE_K / group_size)).to(tl.int32)

    if(W_group_mode >= 2): #[2, 3, 4]
        scales = tl.load(scales_ptr + k_m * stride_meta_g + offs_bn[None, :] * stride_meta_n, eviction_policy=meta_evict_policy) 
    else:
        scales = None
    
    if(W_group_mode == 1 or W_group_mode >= 3): #[1, 3, 4]
        if(zero_is_scalar):
            zeros = tl.load(zeros_ptr, eviction_policy=a_evict)
        else:
            zeros = tl.load(zeros_ptr + k_m * stride_meta_g + offs_bn[None, :] * stride_meta_n, eviction_policy=meta_evict_policy) 
    else:
        zeros = None
    
    if(A_load_order == 2):
        a = tl.load(a_ptrs, eviction_policy=a_evict)

    #tl.join() version
    if(join_version):
        if(elements_per_sample == 2):
            b = tl.join(b, b).permute(0, 2, 1).reshape((BLOCK_SIZE_K, BLOCK_SIZE_N), can_reorder=False) 

        if(elements_per_sample == 8):
            b = tl.join(b, b).permute(0, 2, 1).reshape((BLOCK_SIZE_K // 4, BLOCK_SIZE_N), can_reorder=False) 
            b = tl.join(b, b).permute(0, 2, 1).reshape((BLOCK_SIZE_K // 2, BLOCK_SIZE_N), can_reorder=False) 
            b = tl.join(b, b).permute(0, 2, 1).reshape((BLOCK_SIZE_K, BLOCK_SIZE_N), can_reorder=False)
    ####################################################################
    # Unpack and dequantize
    q_shift = ((offs_k % elements_per_sample) * W_nbits).to(tl.int32)[:, None]
    b = dequantize(b, scales, zeros, q_shift, meta_dtype, unpack_mask, elements_per_sample, W_group_mode, zero_is_scalar)

    if(A_load_order == 3):
        a = tl.load(a_ptrs, eviction_policy=a_evict)

    if(dump_b_val > 0): b = b.to(tl.float32) * dump_b_val

    #Dot product
    if(dot_prod_mode == 0):
        acc = tl.sum(a.reshape((BLOCK_SIZE_K, 1), can_reorder=False).to(acc_dtype) * b.to(acc_dtype), axis=0, keep_dims=True) 
    if(dot_prod_mode == 1):
        acc = tl.sum(a.reshape((BLOCK_SIZE_K, 1), can_reorder=False) * b.to(input_dtype), axis=0, keep_dims=True) 

    if(dump_b_val > 0): acc /= dump_b_val

    ##################################################################
    #Channel-wise scaling
    if(channel_scale_mode == 1): #weight-only
        scales_b = tl.load(scales_ptr + offs_bn, mask=offs_bn < N, other=1, eviction_policy=meta_evict_policy)
        acc      = acc.to(meta_dtype) * scales_b[None, :]

    if(channel_scale_mode == 2): #activation-only
        scales_a = tl.load(scales_a_ptr + offs_am, mask=offs_am < M, other=1, eviction_policy=meta_evict_policy)
        scales_b = tl.full((BLOCK_SIZE_N,), value=1, dtype=meta_dtype)
        acc      = acc.to(meta_dtype) * (scales_a[:, None] * scales_b[None, :])

    if(channel_scale_mode == 3): #weight + activation
        scales_a = tl.load(scales_a_ptr + offs_am, mask=offs_am < M, other=1, eviction_policy=meta_evict_policy)
        scales_b = tl.load(scales_ptr + offs_bn, mask=offs_bn < N,   other=1, eviction_policy=meta_evict_policy)
        acc      = acc.to(meta_dtype) * (scales_a[:, None] * scales_b[None, :])

    ####################################################################
    #Output: tl.atomic_add only supports 1D fp16 arrays, bfp16 would crash 
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_cn = tl.max_contiguous(tl.multiple_of(offs_cn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    c_ptrs  = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    tl.atomic_add(c_ptrs, acc, sem=atomic_mode) 

#TODO: gemv not generating correct reuslts with mxfp dtypes use except for A16W4.
def gemv_forward(x: Tensor, W_q: Tensor, scales: Tensor, zeros: Tensor, scales_x: Tensor,
                                         W_nbits: int, group_size: int, unpack_mask: int, elements_per_sample: int, 
                                         input_dtype: int, output_dtype: int, acc_dtype: int, meta_dtype:int,  
                                         channel_scale_mode: int, W_group_mode: int, data_contiguous: bool, type_id: int,
                                         ) -> Tensor:
    
    global KERNEL_CACHE

    print(elements_per_sample)
    exit()

    M, K, N = x.shape[0], x.shape[1], W_q.shape[1]
    #assert K == W_q.shape[0] * elements_per_sample, "Invalid Input Shapes"

    native_atomic = (output_dtype in [DType.FP16.value, DType.FP32.value]) or NATIVE_ATOMIC
    kernel_output_dtype = (DTYPE_TO_TORCH[output_dtype] if native_atomic else torch.float32)

    if KERNEL.ENABLE_CACHING and M == 1:
        if (M, N) not in KERNEL_CACHE:
            KERNEL_CACHE[(M, N)] = {
                "data": torch.empty((KERNEL.CACHE_SIZE, M, N), device=W_q.device, dtype=kernel_output_dtype),
                "ptr": 0,
            }

        entry = KERNEL_CACHE[(M, N)]
        if entry["ptr"] % KERNEL.CACHE_SIZE == 0:
            entry["data"].zero_()
            entry["ptr"] = 0

        output = entry["data"][entry["ptr"] % KERNEL.CACHE_SIZE]
        entry["ptr"] += 1
        use_prehook = False
    else:
        #output, use_prehook = torch.empty((M, N), device=W_q.device, dtype=kernel_output_dtype), True
        output, use_prehook = torch.zeros((M, N), device=W_q.device, dtype=kernel_output_dtype), False

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']), triton.cdiv(K, meta['BLOCK_SIZE_K']))

    device_index = W_q.device.index

    if(scales_x is not None):
        stride_meta_a_m, stride_meta_a_g = scales_x.stride(0), scales_x.stride(1)
    else:
        stride_meta_a_m, stride_meta_a_g = None, None
        channel_scale_mode = 0

    dtype = DTYPE_TO_TRITON[input_dtype]
    if(dtype in [tl.float16, tl.bfloat16, tl.float32]):
        acc_dtype = dtype
    else:
        acc_dtype = DTYPE_TO_TRITON[acc_dtype]

    if(is_mx_dtype(input_dtype)):
        gemv_kernel = gemv_MX_kernel
        scales = scales.T
        if(scales_x is not None):
            scales_x = scales_x.T
    else:
        gemv_kernel = gemv_INT_kernel

    gemv_kernel[grid](
        x, W_q, output,
        scales, zeros, scales_x,
        fp4_mapping[device_index],
        M, N, K,
        ###########################################
        W_nbits, group_size, unpack_mask, elements_per_sample, type_id, use_prehook,
        x.stride(0), x.stride(1),
        W_q.stride(0), W_q.stride(1),
        output.stride(0), output.stride(1),
        stride_meta_a_m, stride_meta_a_g,
        scales.stride(0), scales.stride(1),
        ############################################
        input_dtype  = DTYPE_TO_TRITON[input_dtype],
        output_dtype = TORCH_DTYPE_TO_TRITON[output.dtype],
        acc_dtype    = acc_dtype,
        meta_dtype   = DTYPE_TO_TRITON[meta_dtype],
        ############################################
        channel_scale_mode = channel_scale_mode,
        W_group_mode       = W_group_mode,
        zero_is_scalar     = zeros.numel() == 1,
        data_contiguous    = data_contiguous,
        dump_b_val         = 0.001 if(W_group_mode in [0, 1] and acc_dtype == DType.FP16.value and W_nbits == 8) else 0, #Warning: Only use with INT8
    )

    if(not native_atomic):
        output = output.to(DTYPE_TO_TORCH[output_dtype])

    return output

def forward_functional(
    x: Tensor,
    bias: Union[None, Tensor],
    tensor_args: List[Tensor],
    meta_args: List[int],
    matmul_type: int = -1, #-1: auto, >=0: manual
) -> Tensor:
    
    data_contiguous = bool(meta_args[1])
    W_nbits = meta_args[1]
    out_features = tensor_args[0].shape[1]

    #Get type_id for autotune: use same autotune for float16/bfloat16
    input_dtype = meta_args[5]
    input_dtype = DType.FP16.value if (input_dtype == DType.BF16.value) else input_dtype
    input_dtype = DType.MXFP16.value if (input_dtype == DType.MXBF16.value) else input_dtype
    type_id = input_dtype * 100 + W_nbits

    if not x.is_contiguous():
        x = x.contiguous()

    batch_size = x.numel() // x.shape[-1]
    orig_shape = x.shape
    out_shape = x.shape[:-1] + (out_features,)
    x_dtype = x.dtype

    scaled_activations = bool(meta_args[0]) and enable_activation_scaling(batch_size)
    #Dynamic activation quantization
    scales_x = None
    if(scaled_activations):
        input_dtype = DType(meta_args[5])
        channel_scale_mode = meta_args[9]

        if(input_dtype in FP8_INT8_DTYPES): #INT8 / FP8
            x, scales_x = scale_activations_per_token(x, w_dtype=DTYPE_TO_TORCH[input_dtype.value])

        elif(input_dtype in [DType.MXFP8] and channel_scale_mode == 4): #MXPF8
            x, scales_x = scale_activations_mxfp8(x, w_dtype=torch.float8_e4m3fn)

        elif(input_dtype in [DType.MXFP8] and channel_scale_mode == 2): #MXPF8 with post-scale for the activations
            x, scales_x = scale_activations_per_token(x, w_dtype=torch.float8_e4m3fn)

        elif(input_dtype in [DType.MXFP4] and channel_scale_mode == 4): #MXPF4
            x, scales_x = scale_activations_mxfp4(x)

        elif(input_dtype in [DType.NVFP4] and channel_scale_mode == 4): #NVPF4: TODO
            x, scales_x = scale_activations_nvfp4(x)
    
    x = x.view(-1, x.shape[-1])

    if(matmul_type >= 0):
        matmul_type_str = GEMLITE_MATMUL_TYPES[matmul_type]
    else:
        matmul_type_str = get_matmul_type(x.shape[0], W_nbits, is_mx_dtype(input_dtype)) #batch_size, W_nbits, is_mx_dtype

    out = (
        GEMLITE_TRITON_MAPPING[matmul_type_str]
        .forward(
            x, *tensor_args, scales_x, *meta_args[1:-1], data_contiguous, type_id
        )
        .view(out_shape)
    )

    if bias is not None:
        out += bias

    return out