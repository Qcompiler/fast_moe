


def forward(q_x, act_scale, )
    q_x, act_scale = a8_per_token_act_quant(x)
    out = w4a8_gemm_per_token_per_channel_asymm(
        q_x,
        act_scale,
        self.get_native_layout_qweight(),
        self.s1_scales,
        self.s1_szeros,
    )
