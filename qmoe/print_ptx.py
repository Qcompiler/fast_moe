import re
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple
import networkx as nx
import matplotlib.pyplot as plt

@dataclass
class Register:
    """寄存器信息"""
    name: str
    reg_type: str
    size: int  # 位宽
    
@dataclass
class Instruction:
    """指令信息"""
    opcode: str
    operands: List[str]
    line_no: int
    dest_reg: str = ""  # 目标寄存器
    
    def __str__(self):
        return f"{self.opcode} {' '.join(self.operands)}"

class EnhancedAssemblyParser:
    """增强版汇编解析器（带数据流分析）"""
    
    def __init__(self, asm_code: str):
        self.asm_code = asm_code
        self.registers: Dict[str, Register] = {}
        self.instructions: List[Instruction] = []
        self.data_flow_graph = nx.DiGraph()
        
    def parse(self):
        """解析汇编代码"""
        self._extract_and_parse_registers()
        self._parse_instructions()
        self._build_data_flow_graph()
        return self
    
    def _extract_and_parse_registers(self):
        """提取和解析寄存器声明"""
        # 查找所有寄存器声明
        reg_pattern = r'\.reg\s+(\.\w+)\s+([^;]+);'
        matches = re.findall(reg_pattern, self.asm_code)
        
        for reg_type, reg_names in matches:
            # 解析位宽
            size_map = {
                '.b8': 8, '.b16': 16, '.b32': 32, '.b64': 64,
                '.u8': 8, '.u16': 16, '.u32': 32, '.u64': 64,
                '.s8': 8, '.s16': 16, '.s32': 32, '.s64': 64,
                '.f16': 16, '.f32': 32, '.f64': 64
            }
            size = size_map.get(reg_type, 32)
            
            # 处理寄存器列表
            for reg_spec in reg_names.split(','):
                reg_spec = reg_spec.strip()
                
                # 处理寄存器组（如r<16>）
                if '<' in reg_spec:
                    base_name, count = reg_spec.split('<')
                    count = int(count.replace('>', ''))
                    
                    for i in range(count):
                        reg_name = f"{base_name}{i}"
                        self.registers[reg_name] = Register(reg_name, reg_type, size)
                else:
                    self.registers[reg_spec] = Register(reg_spec, reg_type, size)
    
    def _parse_instructions(self):
        """解析指令序列"""
        # 提取指令部分
        code = self.asm_code.strip()
        code = code[code.find('{')+1:code.rfind('}')]
        
        # 按行分割，过滤空行和寄存器声明
        lines = [line.strip() for line in code.split('\n') 
                if line.strip() and not line.strip().startswith('.reg')]
        
        for i, line in enumerate(lines):
            # 清理行（移除多余空格和逗号）
            line = re.sub(r'\s+', ' ', line)
            line = line.replace(',', ' ')
            
            # 分割指令和操作数
            parts = line.split()
            if not parts:
                continue
                
            opcode = parts[0]
            operands = parts[1:] if len(parts) > 1 else []
            
            # 提取目标寄存器
            dest_reg = ""
            if operands:
                # 对于大多数指令，第一个操作数是目标寄存器
                if opcode.startswith(('mov', 'lop3', 'fma', 'sub', 'add', 'mul', 'shr', 'shl')):
                    dest_reg = operands[0]
                elif opcode.startswith(('cvt', 'and', 'or', 'xor')):
                    dest_reg = operands[0]
            
            instr = Instruction(opcode, operands, i, dest_reg)
            self.instructions.append(instr)
            
            # 添加到图
            self.data_flow_graph.add_node(i, instruction=str(instr))
    
    def _build_data_flow_graph(self):
        """构建数据流图"""
        # 追踪寄存器的最后一次写入位置
        last_write: Dict[str, int] = {}
        
        for i, instr in enumerate(self.instructions):
            # 查找数据依赖
            for op in instr.operands:
                if op in self.registers or re.match(r'^r\d+$', op) or re.match(r'^tmp\d+$', op):
                    # 这个操作数是一个寄存器
                    if op in last_write:
                        # 添加从最后一次写入到当前读取的边
                        self.data_flow_graph.add_edge(last_write[op], i,
                                                      label=f"reg:{op}")
            
            # 更新最后一次写入位置
            if instr.dest_reg:
                last_write[instr.dest_reg] = i
    
    def visualize_data_flow(self):
        """可视化数据流图"""
        plt.figure(figsize=(15, 10))
        
        # 使用分层布局
        pos = nx.spring_layout(self.data_flow_graph, k=2, iterations=50)
        
        # 绘制节点
        nx.draw_networkx_nodes(self.data_flow_graph, pos, 
                              node_color='lightblue',
                              node_size=3000,
                              alpha=0.8)
        
        # 绘制边
        nx.draw_networkx_edges(self.data_flow_graph, pos,
                              edge_color='gray',
                              arrows=True,
                              arrowsize=20)
        
        # 绘制标签
        labels = nx.get_node_attributes(self.data_flow_graph, 'instruction')
        nx.draw_networkx_labels(self.data_flow_graph, pos, labels, font_size=8)
        
        # 绘制边标签
        edge_labels = nx.get_edge_attributes(self.data_flow_graph, 'label')
        nx.draw_networkx_edge_labels(self.data_flow_graph, pos, 
                                    edge_labels=edge_labels,
                                    font_size=7)
        
        plt.title("Assembly Data Flow Graph", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('data_flow_graph.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_pipeline(self):
        """可视化指令流水线"""
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # 1. 指令时间线
        ax1 = axes[0]
        for i, instr in enumerate(self.instructions):
            # 根据指令类型着色
            color = 'lightblue'
            if 'mov' in instr.opcode:
                color = 'lightgreen'
            elif 'lop3' in instr.opcode:
                color = 'lightcoral'
            elif 'fma' in instr.opcode:
                color = 'orange'
            elif any(op in instr.opcode for op in ['sub', 'add', 'mul']):
                color = 'lightyellow'
            elif any(op in instr.opcode for op in ['shr', 'shl', 'and', 'cvt']):
                color = 'lightpink'
            
            ax1.barh(i, 1, color=color, edgecolor='black')
            ax1.text(0.5, i, str(instr), ha='center', va='center', fontsize=8)
        
        ax1.set_yticks(range(len(self.instructions)))
        ax1.set_yticklabels([f"L{i}" for i in range(len(self.instructions))])
        ax1.set_xlabel('Instruction Pipeline')
        ax1.set_title('Instruction Sequence Timeline')
        ax1.invert_yaxis()
        
        # 2. 寄存器使用热图
        ax2 = axes[1]
        
        # 收集所有寄存器
        all_regs = sorted(list(self.registers.keys()))
        usage_matrix = []
        
        for instr in self.instructions:
            row = []
            for reg in all_regs:
                # 检查寄存器是否在指令中使用
                used = any(reg in op for op in instr.operands) or reg == instr.dest_reg
                row.append(1 if used else 0)
            usage_matrix.append(row)
        
        im = ax2.imshow(usage_matrix, cmap='YlOrRd', aspect='auto')
        ax2.set_xticks(range(len(all_regs)))
        ax2.set_xticklabels(all_regs, rotation=45, ha='right', fontsize=8)
        ax2.set_yticks(range(len(self.instructions)))
        ax2.set_yticklabels([f"L{i}" for i in range(len(self.instructions))])
        ax2.set_title('Register Usage Heatmap')
        ax2.set_xlabel('Registers')
        ax2.set_ylabel('Instruction Line')
        
        plt.colorbar(im, ax=ax2)
        plt.tight_layout()
        plt.savefig('instruction_pipeline.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_analysis(self):
        """打印分析结果"""
        print("=" * 80)
        print("汇编代码分析报告")
        print("=" * 80)
        
        print(f"\n1. 寄存器统计:")
        print(f"   总共 {len(self.registers)} 个寄存器")
        
        # 按类型统计
        type_count = {}
        for reg in self.registers.values():
            type_count[reg.reg_type] = type_count.get(reg.reg_type, 0) + 1
        
        for reg_type, count in type_count.items():
            print(f"   {reg_type}: {count} 个")
        
        print(f"\n2. 指令统计:")
        print(f"   总共 {len(self.instructions)} 条指令")
        
        # 按操作码统计
        opcode_count = {}
        for instr in self.instructions:
            opcode_count[instr.opcode] = opcode_count.get(instr.opcode, 0) + 1
        
        for opcode, count in sorted(opcode_count.items()):
            print(f"   {opcode}: {count} 次")
        
        print(f"\n3. 数据流分析:")
        print(f"   数据流图包含 {self.data_flow_graph.number_of_nodes()} 个节点")
        print(f"   数据流图包含 {self.data_flow_graph.number_of_edges()} 条边")
        
        # 计算关键路径
        try:
            longest_path = nx.dag_longest_path(self.data_flow_graph)
            print(f"   关键路径长度: {len(longest_path)} 条指令")
        except:
            print("   无法计算关键路径（可能存在循环）")

# 主程序
if __name__ == "__main__":
    asm_code = """
    {
    .reg .b32 	r<16>;
    .reg .b32  r_high<2>, r_low<2>;

      .reg .b64 	rd<2>;
    .reg .u16 tmp1, tmp2, tmp3, tmp4;
    
    mov.u32 r2, $2;
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
    cvt.u16.u32   tmp1, r_high1;
    and.b32       r_low1, r9, 0xFFFF;
    cvt.u16.u32   tmp2, r_low1;
    shr.s32   r_high1, r12, 16;
    cvt.u16.u32   tmp3, r_high1;
    and.b32       r_low1, r12, 0xFFFF;
    cvt.u16.u32   tmp4, r_low1;
    mov.b32 $1, {tmp4, tmp3};   
    mov.b32 $0, {tmp2, tmp1};  
    }
    """
    
    print("开始解析汇编代码...")
    
    # 使用增强版解析器
    parser = EnhancedAssemblyParser(asm_code)
    analysis = parser.parse()
    
    # 打印分析报告
    analysis.print_analysis()
    
    # 可视化
    print("\n生成可视化图表...")
    analysis.visualize_data_flow()
    analysis.visualize_pipeline()
    
    print("\n完成！已生成以下文件:")
    print("1. data_flow_graph.png - 数据流图")
    print("2. instruction_pipeline.png - 指令流水线图")