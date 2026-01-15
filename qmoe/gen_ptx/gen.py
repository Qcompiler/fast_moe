import re
from typing import Dict, List, Tuple

class PTXtoTritonConverter:
    def __init__(self, ptx_code: str):
        self.ptx_code = ptx_code
        self.registers = {}
        self.constants = {}
        self.instructions = []
        
    def parse_ptx(self):
        """解析PTX代码"""
        lines = self.ptx_code.strip().split('\n')
        for line in lines:
            line = line.strip()
            
            # 跳过空行和注释
            if not line or line.startswith('//'):
                continue
                
            # 提取指令
            if line.startswith('ld.param') or line.startswith('mov.u32') or line.startswith('shr.u32'):
                self.process_instruction(line)
            elif 'lop3.b32' in line or 'sub.f16x2' in line or 'fma.rn.f16x2' in line or 'mov.b32' in line:
                self.process_instruction(line)
            elif 'st.param' in line or line.startswith('ret'):
                continue  # 跳过存储和返回指令
    
    def process_instruction(self, line: str):
        """处理单条指令"""
        # 清理内联汇编标记
        line = line.replace('// begin inline asm', '').replace('// end inline asm', '').strip()
        
        # 处理各种指令
        if 'ld.param.u32' in line:
            # 输入参数加载
            match = re.search(r'%r(\d+),\s*\[dequant_param_0\]', line)
            if match:
                self.registers[match.group(1)] = 'input'
        elif 'shr.u32' in line:
            match = re.search(r'shr\.u32\s+%r(\d+),\s*%r(\d+),\s*(\d+)', line)
            if match:
                self.instructions.append(f'shr.u32 r{match.group(1)}, r{match.group(2)}, {match.group(3)};')
        elif 'lop3.b32' in line:
            match = re.search(r'lop3\.b32\s+%r(\d+),\s*%r(\d+),\s*(\d+),\s*(\d+),\s*(\d+)', line)
            if match:
                self.instructions.append(f'lop3.b32 r{match.group(1)}, r{match.group(2)}, {match.group(3)}, {match.group(4)}, {match.group(5)};')
        elif 'sub.f16x2' in line:
            match = re.search(r'sub\.f16x2\s+%r(\d+),\s*%r(\d+),\s*%r(\d+)', line)
            if match:
                self.instructions.append(f'sub.f16x2 r{match.group(1)}, r{match.group(2)}, r{match.group(3)};')
        elif 'fma.rn.f16x2' in line:
            match = re.search(r'fma\.rn\.f16x2\s+%r(\d+),\s*%r(\d+),\s*%r(\d+),\s*%r(\d+)', line)
            if match:
                self.instructions.append(f'fma.rn.f16x2 r{match.group(1)}, r{match.group(2)}, r{match.group(3)}, r{match.group(4)};')
        elif 'mov.u32' in line:
            match = re.search(r'mov\.u32\s+%r(\d+),\s*([-\d]+)', line)
            if match:
                const_val = int(match.group(2))
                if const_val < 0:
                    # 处理负数常量
                    const_val = self.to_twos_complement(const_val)
                self.instructions.append(f'mov.u32 r{match.group(1)}, {const_val};')
                self.constants[match.group(1)] = const_val
        elif 'mov.b32' in line:
            match = re.search(r'mov\.b32\s+%f(\d+),\s*%r(\d+)', line)
            if match:
                self.instructions.append(f'mov.b32 f{match.group(1)}, r{match.group(2)};')
    
    def to_twos_complement(self, value: int, bits=32) -> int:
        """将负数转换为补码表示"""
        if value >= 0:
            return value
        return (1 << bits) + value
    
    def generate_triton_asm(self) -> str:
        """生成Triton内联汇编代码"""
        # 构建寄存器映射
        reg_mapping = {}
        
        # 生成汇编代码块
        asm_lines = [
            ".reg .b32  r<23>;",
            ".reg .f32  f<5>;",
            "",
            "mov.u32  r2, $4;",  # $4 对应输入参数
            "shr.u32  r8, r2, 8;"
        ]
        
        # 添加所有指令
        for instr in self.instructions:
            asm_lines.append(f"    {instr}")
        
        # 添加输出部分 - 将f寄存器的值移动到输出参数
        asm_lines.extend([
            "",
            "mov.b32   $0, f4;",  # 第一个输出
            "mov.b32   $1, f2;",  # 第二个输出  
            "mov.b32   $2, f3;",  # 第三个输出
            "mov.b32   $3, f1;",  # 第四个输出
        ])
        
        return "\n".join(asm_lines)
    
    def generate_python_code(self) -> str:
        """生成完整的Python函数代码"""
        asm_code = self.generate_triton_asm()
        
        python_code = f'''@triton.jit
def dequanti_tensorRT_llm(b):
    """
    将int32的4-bit量化值反量化为4个uint32（包含8个half）
    
    Args:
        b: 包含8个4-bit量化值的int32
        
    Returns:
        4个uint32，每个包含2个half
    """
    x1, x2, x3, x4 = tl.inline_asm_elementwise(
        asm=\"\"\"{{
            {asm_code}
        }}\"\"\",
        constraints=(
            "=r,=r,=r,=r,r"
        ),
        args=[b],
        dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
        is_pure=False,
        pack=1,
    )
    return x1, x2, x3, x4


'''
        return python_code

def main():
    # 你的PTX代码
    f = open("i4_kernel.ptx")
    ptx_code = "".join(f.readlines())

    print(ptx_code)
    # exit()

    # 创建转换器
    converter = PTXtoTritonConverter(ptx_code)
    converter.parse_ptx()
    
    # 生成Python代码
    python_output = converter.generate_python_code()
    
    # 输出结果
    print("生成的Triton Python代码:")
    print("=" * 80)
    print(python_output)
    
    # 也可以保存到文件
    with open("dequant_triton.py", "w") as f:
        f.write(python_output)
    
    print("\n代码已保存到 dequant_triton.py")

if __name__ == "__main__":
    main()
