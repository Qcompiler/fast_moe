import ast
import graphviz
from typing import List, Dict, Any
import re

class AssemblyASTNode:
    """汇编AST节点基类"""
    def __init__(self, node_type: str, value: str = "", children: List = None):
        self.node_type = node_type
        self.value = value
        self.children = children or []
    
    def add_child(self, child):
        self.children.append(child)
        return self
    
    def __repr__(self):
        return f"{self.node_type}({self.value})"

class AssemblyParser:
    """汇编代码解析器"""
    
    def __init__(self, asm_code: str):
        self.asm_code = asm_code
        self.root = AssemblyASTNode("InlineAssemblyBlock", "PTX Inline Assembly")
        
    def parse(self):
        """解析汇编代码"""
        # 解析寄存器声明
        self._parse_registers()
        
        # 解析指令序列
        self._parse_instructions()
        
        return self.root
    
    def _parse_registers(self):
        """解析寄存器声明部分"""
        reg_pattern = r'\.reg\s+(\.\w+)\s+([^;]+);'
        matches = re.findall(reg_pattern, self.asm_code)
        
        reg_decl_node = AssemblyASTNode("RegisterDeclarations", "Register Declarations")
        
        for reg_type, reg_names in matches:
            # 处理寄存器组（如r<16>）
            if '<' in reg_names:
                # print(reg_names.split('<'))
                 
                name, count = reg_names.split('<')
                count = count.replace('>', '').strip()
                reg_node = AssemblyASTNode("RegisterGroup", 
                                          f"{name}[0-{int(count)-1}]")
                reg_node.add_child(AssemblyASTNode("RegisterType", reg_type))
            else:
                # 处理多个寄存器声明
                for reg in reg_names.split(','):
                    reg = reg.strip()
                    reg_node = AssemblyASTNode("Register", reg)
                    reg_node.add_child(AssemblyASTNode("RegisterType", reg_type))
                    reg_decl_node.add_child(reg_node)
        
        self.root.add_child(reg_decl_node)
    
    def _parse_instructions(self):
        """解析指令序列"""
        # 提取指令部分（移除花括号和寄存器声明）
        code = self.asm_code.strip()
        code = code[code.find('{')+1:code.rfind('}')]
        
        # 移除寄存器声明行
        lines = [line.strip() for line in code.split('\n') 
                if line.strip() and not line.strip().startswith('.reg')]
        
        instructions_node = AssemblyASTNode("MicroKernel", "Dequant")
        
        for line in lines:
            if not line:
                continue
                
            # 解析指令
            if '=' in line and line.strip().startswith('mov'):
                # 输出指令
                parts = line.replace(',', ' ').split()
                if len(parts) >= 3:
                    instr_node = AssemblyASTNode("OutputInstruction", parts[0])
                    instr_node.add_child(AssemblyASTNode("Destination", parts[1]))
                    instr_node.add_child(AssemblyASTNode("Source", ' '.join(parts[2:])))
                    instructions_node.add_child(instr_node)
            elif any(op in line for op in ['mov', 'lop3', 'fma', 'sub', 'shr', 'cvt', 'and']):
                # 普通指令
                parts = line.replace(',', ' ').split()
                if parts:
                    instr_node = AssemblyASTNode("Instruction", parts[0])
                    for part in parts[1:]:
                        if part:
                            instr_node.add_child(AssemblyASTNode("Operand", part))
                    instructions_node.add_child(instr_node)
        
        self.root.add_child(instructions_node)
    
    def visualize_ast(self, output_file="assembly_ast"):
        """可视化AST"""
        dot = graphviz.Digraph(comment='Assembly AST', format='png')
        dot.attr(rankdir='TB')
        
        # 为不同类型的节点定义样式
        node_styles = {
            "InlineAssemblyBlock": {"color": "lightpink", "style": "filled", "shape": "ellipse"},
            "RegisterDeclarations": {"color": "lightblue", "style": "filled", "shape": "ellipse"},
            "RegisterGroup": {"color": "lightblue", "style": "filled", "shape": "ellipse"},
            "Register": {"color": "lightblue", "style": "filled", "shape": "ellipse"},
            "MicroKernel": {"color": "lightgreen", "style": "filled", "shape": "ellipse"},
            "Instruction": {"color": "lightyellow", "style": "filled", "shape": "ellipse"},
            "OutputInstruction": {"color": "orange", "style": "filled", "shape": "ellipse"},
            "Destination": {"color": "lightcoral", "style": "filled", "shape": "ellipse"},
            "Source": {"color": "lightcyan", "style": "filled", "shape": "ellipse"},
            "Operand": {"color": "lightgray", "style": "filled", "shape": "ellipse"},
            "RegisterType": {"color": "lightblue", "style": "filled", "shape": "ellipse"},
        }
        
        def add_node(node, parent_id=None):
            """递归添加节点"""
            node_id = str(id(node))
             
            
            # 确定节点样式
            style = node_styles.get(node.node_type, 
                                   {"color": "white", "style": "filled", "shape": "box"})
            
            # 创建节点标签
            if node.value:
                label = f"{node.node_type}\n{node.value}"
            else:
                label = node.node_type
            
            if "Register" in label:
                return

            # if not "Inline" in label:
            dot.node(node_id, label, 
                    color=style["color"], 
                    style=style["style"], 
                    shape=style["shape"])
        
            if parent_id:
                dot.edge(parent_id, node_id)
            
            # 添加子节点
            for child in node.children:
                add_node(child, node_id)
        
        # 添加所有节点
        add_node(self.root.children[1])
        
        # 保存和显示
        dot.render(output_file, format = "png", cleanup=True)
        return dot

# 测试代码
if __name__ == "__main__":
    asm_code = """
    {
    .reg .b32 	r<16>;
    .reg .b64 	rd<2>;
    mov.u32 r2, $2;
    mov.u32 	r3, 983055;
    mov.u32 	r8, 1677747200;
    lop3.b32 r1, r2, r3, r8, 234;
    mov.u32 	r7, 15728880;
    lop3.b32 r5, r2, r7, r8, 234;

    }
    """
    
    # 创建解析器
    parser = AssemblyParser(asm_code)
    ast_root = parser.parse()
    
    # 可视化AST
    print("开始生成AST可视化...")
    
    dot = parser.visualize_ast("assembly_ast_detailed")
    
    print(f"AST已保存为 assembly_ast_detailed.png")
    
    # 打印AST文本表示
    # def print_ast(node, indent=0):
    #     print("  " * indent + str(node))
    #     for child in node.children:
    #         print_ast(child, indent + 1)
    
    # print("\nAST文本表示:")
    # print_ast(ast_root)