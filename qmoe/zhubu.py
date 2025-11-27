@triton.jit
def dequanti(b):
    x1, x2, x3, x4 = tl.inline_asm_elementwise(
        asm="""
            {
            .reg .b32 	r<16>;
            .reg .b32  r_high<2>, r_low<2>;

	        .reg .b64 	rd<2>;
            mov.u32 r2, $4;
            mov.u32 	r3, 983055;
            mov.u32 	r8, 1677747200;
            lop3.b32 r1, r2, r3, r8, 234;
            mov.u32 	r7, 15728880;
            lop3.b32 r5, r2, r7, r8, 234;
            mov.u32 	r11, 1678271496;
            mov.u32 	r14, 738208768;
            mov.u32 	r15, -729754496;
            fma.rn.f16x2 r12,r5,r14,r15;
            sub.f16x2 r9,r1,r11;
            shr.s32   r_high1, r9, 16;
            cvt.u16.u32   $0, r_high1;
            and.b32       r_low1, r9, 0xFFFF;
            cvt.u16.u32   $1, r_low1;
            shr.s32   r_high1, r12, 16;
            cvt.u16.u32   $2, r_high1;
            and.b32       r_low1, r12, 0xFFFF;
            cvt.u16.u32   $3, r_low1;

            }
        """,
        constraints=(
            "=f,=f,=f,=f,r"
        ),
        args=[b], #输入
        dtype=(tl.float16, tl.float16, tl.float16, tl.float16), #输出
        is_pure=False,
        pack=1,
    )

    
    return x1, x2, x3, x4