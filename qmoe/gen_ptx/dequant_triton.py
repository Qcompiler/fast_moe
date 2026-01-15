@triton.jit
def dequanti_tensorRT_llm(b):
    """
    将int32的4-bit量化值反量化为4个uint32（包含8个half）
    
    Args:
        b: 包含8个4-bit量化值的int32
        
    Returns:
        4个uint32，每个包含2个half
    """
    x1, x2, x3, x4 = tl.inline_asm_elementwise(
        asm="""{
        .reg .b32  r<23>;
        .reg .f32  f<5>;

        mov.u32  r2, $4;
        shr.u32  r8, r2, 8;
            shr.u32 r8, r2, 8;
            lop3.b32 r1, r2, 983055, 1677747200, 234;
            lop3.b32 r3, r2, 15728880, 1677747200, 234;
            lop3.b32 r5, r8, 983055, 1677747200, 234;
            lop3.b32 r7, r8, 15728880, 1677747200, 234;
            mov.u32 r18, 1678271496;
            sub.f16x2 r9, r1, r18;
            mov.u32 r21, 738208768;
            mov.u32 r22, 3565212800;
            fma.rn.f16x2 r12, r3, r21, r22;
            sub.f16x2 r16, r5, r18;
            fma.rn.f16x2 r19, r7, r21, r22;
            mov.b32 f1, r19;
            mov.b32 f2, r12;
            mov.b32 f3, r16;
            mov.b32 f4, r9;

        mov.b32   $0, f4;
        mov.b32   $1, f2;
        mov.b32   $2, f3;
        mov.b32   $3, f1;
        }""",
        constraints=(
            "=r,=r,=r,=r,r"
        ),
        args=[b],
        dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
        is_pure=False,
        pack=1,
    )
    return x1, x2, x3, x4


