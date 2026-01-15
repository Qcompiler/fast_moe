
# __CLANG_OFFLOAD_BUNDLE____START__ hip-amdgcn-amd-amdhsa--gfx90a
	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx90a"
	.amdhsa_code_object_version 6
	.protected	_Z5main_P15HIP_vector_typeIfLj4EEPKi ; -- Begin function _Z5main_P15HIP_vector_typeIfLj4EEPKi
	.globl	_Z5main_P15HIP_vector_typeIfLj4EEPKi
	.p2align	8
	.type	_Z5main_P15HIP_vector_typeIfLj4EEPKi,@function
_Z5main_P15HIP_vector_typeIfLj4EEPKi:   ; @_Z5main_P15HIP_vector_typeIfLj4EEPKi
; %bb.0:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z5main_P15HIP_vector_typeIfLj4EEPKi
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 6
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 0
		.amdhsa_accum_offset 4
		.amdhsa_reserve_vcc 0
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	_Z5main_P15HIP_vector_typeIfLj4EEPKi, .Lfunc_end0-_Z5main_P15HIP_vector_typeIfLj4EEPKi
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 4
; NumSgprs: 4
; NumVgprs: 0
; NumAgprs: 0
; TotalNumVgprs: 0
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 4
; NumVGPRsForWavesPerEU: 1
; AccumOffset: 4
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 0
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.type	__hip_cuid_c7b42489f38a2b2a,@object ; @__hip_cuid_c7b42489f38a2b2a
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_c7b42489f38a2b2a
__hip_cuid_c7b42489f38a2b2a:
	.byte	0                               ; 0x0
	.size	__hip_cuid_c7b42489f38a2b2a, 1

	.ident	"AMD clang version 19.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.4.1 25184 c87081df219c42dc27c5b6d86c0525bc7d01f727)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_c7b42489f38a2b2a
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z5main_P15HIP_vector_typeIfLj4EEPKi
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .sgpr_spill_count: 0
    .symbol:         _Z5main_P15HIP_vector_typeIfLj4EEPKi.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     0
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx90a
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata

# __CLANG_OFFLOAD_BUNDLE____END__ hip-amdgcn-amd-amdhsa--gfx90a

# __CLANG_OFFLOAD_BUNDLE____START__ host-x86_64-unknown-linux-gnu-
	.text
	.file	"amdsum.cpp"
	.globl	_Z20__device_stub__main_P15HIP_vector_typeIfLj4EEPKi # -- Begin function _Z20__device_stub__main_P15HIP_vector_typeIfLj4EEPKi
	.p2align	4, 0x90
	.type	_Z20__device_stub__main_P15HIP_vector_typeIfLj4EEPKi,@function
_Z20__device_stub__main_P15HIP_vector_typeIfLj4EEPKi: # @_Z20__device_stub__main_P15HIP_vector_typeIfLj4EEPKi
	.cfi_startproc
# %bb.0:
	subq	$88, %rsp
	.cfi_def_cfa_offset 96
	movq	%rdi, 56(%rsp)
	movq	%rsi, 48(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 64(%rsp)
	leaq	48(%rsp), %rax
	movq	%rax, 72(%rsp)
	leaq	32(%rsp), %rdi
	leaq	16(%rsp), %rsi
	leaq	8(%rsp), %rdx
	movq	%rsp, %rcx
	callq	__hipPopCallConfiguration
	movq	32(%rsp), %rsi
	movl	40(%rsp), %edx
	movq	16(%rsp), %rcx
	movl	24(%rsp), %r8d
	leaq	64(%rsp), %r9
	movl	$_Z5main_P15HIP_vector_typeIfLj4EEPKi, %edi
	pushq	(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$104, %rsp
	.cfi_adjust_cfa_offset -104
	retq
.Lfunc_end0:
	.size	_Z20__device_stub__main_P15HIP_vector_typeIfLj4EEPKi, .Lfunc_end0-_Z20__device_stub__main_P15HIP_vector_typeIfLj4EEPKi
	.cfi_endproc
                                        # -- End function
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:
	subq	$88, %rsp
	.cfi_def_cfa_offset 96
	movabsq	$4294967297, %rdi               # imm = 0x100000001
	movl	$1, %esi
	movq	%rdi, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration
	testl	%eax, %eax
	jne	.LBB1_2
# %bb.1:
	movq	$0, 56(%rsp)
	movq	$0, 48(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 64(%rsp)
	leaq	48(%rsp), %rax
	movq	%rax, 72(%rsp)
	leaq	32(%rsp), %rdi
	leaq	16(%rsp), %rsi
	leaq	8(%rsp), %rdx
	movq	%rsp, %rcx
	callq	__hipPopCallConfiguration
	movq	32(%rsp), %rsi
	movl	40(%rsp), %edx
	movq	16(%rsp), %rcx
	movl	24(%rsp), %r8d
	leaq	64(%rsp), %r9
	movl	$_Z5main_P15HIP_vector_typeIfLj4EEPKi, %edi
	pushq	(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB1_2:
	xorl	%eax, %eax
	addq	$88, %rsp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90                         # -- Begin function __hip_module_ctor
	.type	__hip_module_ctor,@function
__hip_module_ctor:                      # @__hip_module_ctor
	.cfi_startproc
# %bb.0:
	subq	$40, %rsp
	.cfi_def_cfa_offset 48
	cmpq	$0, __hip_gpubin_handle_c7b42489f38a2b2a(%rip)
	jne	.LBB2_2
# %bb.1:
	movl	$__hip_fatbin_wrapper, %edi
	callq	__hipRegisterFatBinary
	movq	%rax, __hip_gpubin_handle_c7b42489f38a2b2a(%rip)
.LBB2_2:
	movq	__hip_gpubin_handle_c7b42489f38a2b2a(%rip), %rdi
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$_Z5main_P15HIP_vector_typeIfLj4EEPKi, %esi
	movl	$.L__unnamed_1, %edx
	movl	$.L__unnamed_1, %ecx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	movl	$__hip_module_dtor, %edi
	addq	$40, %rsp
	.cfi_def_cfa_offset 8
	jmp	atexit                          # TAILCALL
.Lfunc_end2:
	.size	__hip_module_ctor, .Lfunc_end2-__hip_module_ctor
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90                         # -- Begin function __hip_module_dtor
	.type	__hip_module_dtor,@function
__hip_module_dtor:                      # @__hip_module_dtor
	.cfi_startproc
# %bb.0:
	movq	__hip_gpubin_handle_c7b42489f38a2b2a(%rip), %rdi
	testq	%rdi, %rdi
	je	.LBB3_2
# %bb.1:
	pushq	%rax
	.cfi_def_cfa_offset 16
	callq	__hipUnregisterFatBinary
	movq	$0, __hip_gpubin_handle_c7b42489f38a2b2a(%rip)
	addq	$8, %rsp
	.cfi_def_cfa_offset 8
.LBB3_2:
	retq
.Lfunc_end3:
	.size	__hip_module_dtor, .Lfunc_end3-__hip_module_dtor
	.cfi_endproc
                                        # -- End function
	.type	_Z5main_P15HIP_vector_typeIfLj4EEPKi,@object # @_Z5main_P15HIP_vector_typeIfLj4EEPKi
	.section	.rodata,"a",@progbits
	.globl	_Z5main_P15HIP_vector_typeIfLj4EEPKi
	.p2align	3, 0x0
_Z5main_P15HIP_vector_typeIfLj4EEPKi:
	.quad	_Z20__device_stub__main_P15HIP_vector_typeIfLj4EEPKi
	.size	_Z5main_P15HIP_vector_typeIfLj4EEPKi, 8

	.type	.L__unnamed_1,@object           # @0
	.section	.rodata.str1.1,"aMS",@progbits,1
.L__unnamed_1:
	.asciz	"_Z5main_P15HIP_vector_typeIfLj4EEPKi"
	.size	.L__unnamed_1, 37

	.type	__hip_fatbin_wrapper,@object    # @__hip_fatbin_wrapper
	.section	.hipFatBinSegment,"a",@progbits
	.p2align	3, 0x0
__hip_fatbin_wrapper:
	.long	1212764230                      # 0x48495046
	.long	1                               # 0x1
	.quad	__hip_fatbin_c7b42489f38a2b2a
	.quad	0
	.size	__hip_fatbin_wrapper, 24

	.type	__hip_gpubin_handle_c7b42489f38a2b2a,@object # @__hip_gpubin_handle_c7b42489f38a2b2a
	.local	__hip_gpubin_handle_c7b42489f38a2b2a
	.comm	__hip_gpubin_handle_c7b42489f38a2b2a,8,8
	.section	.init_array,"aw",@init_array
	.p2align	3, 0x0
	.quad	__hip_module_ctor
	.type	__hip_cuid_c7b42489f38a2b2a,@object # @__hip_cuid_c7b42489f38a2b2a
	.bss
	.globl	__hip_cuid_c7b42489f38a2b2a
__hip_cuid_c7b42489f38a2b2a:
	.byte	0                               # 0x0
	.size	__hip_cuid_c7b42489f38a2b2a, 1

	.ident	"AMD clang version 19.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.4.1 25184 c87081df219c42dc27c5b6d86c0525bc7d01f727)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _Z20__device_stub__main_P15HIP_vector_typeIfLj4EEPKi
	.addrsig_sym __hip_module_ctor
	.addrsig_sym __hip_module_dtor
	.addrsig_sym _Z5main_P15HIP_vector_typeIfLj4EEPKi
	.addrsig_sym __hip_fatbin_c7b42489f38a2b2a
	.addrsig_sym __hip_fatbin_wrapper
	.addrsig_sym __hip_cuid_c7b42489f38a2b2a

# __CLANG_OFFLOAD_BUNDLE____END__ host-x86_64-unknown-linux-gnu-
