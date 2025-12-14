import ast
import inspect
import triton
import triton.language as tl

@triton.jit
 
def gemv_int4_kernel(
    A_ptr, x_ptr, y_ptr,
    m, k,
    stride_am, stride_an,
    BLOCK_SIZE: tl.constexpr,
    unpack_mask: tl.constexpr,
    a_evict: tl.constexpr = 'evict_last',
    b_evict: tl.constexpr = 'evict_first',
):
    row_id = tl.program_id(0)
    acc = 0.0
    elements_per_sample = 8
    W_nbits = 4
    offs_k =  tl.arange(0, BLOCK_SIZE)
    A_offset = row_id * stride_am + (offs_k // 8) * stride_an
    A_ptr = A_ptr + A_offset
    x_ptr = x_ptr + (offs_k)
    for kk in range(0, tl.cdiv(k, BLOCK_SIZE)):
      a = tl.load(A_ptr,  eviction_policy=a_evict)
 
      a = ((a >> ((offs_k % elements_per_sample) * W_nbits).to(tl.int32)) & unpack_mask) - 8 
      a = a.to(tl.float16)


      x = tl.load(x_ptr)
      acc += tl.sum(a * x, axis=0)

    tl.store(y_ptr + row_id, acc)


@triton.jit
def dequanti(b):
    x1, x2 = tl.inline_asm_elementwise(
        asm="""
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
        """,
        constraints=(
            "=r,=r,r"
        ),
        args=[b], #输入
        dtype=(tl.uint32, tl.uint32), #输出
        is_pure=False,
        pack=1,
    )

    
    return x1, x2




@triton.jit
def sum_4_half(x1, x2, vec1, vec2):
    # x1 : int32 x2: int32
    # vec1 : int32 vec2: int32
    # x1 * vec1 + x2 * vec2
    y = tl.inline_asm_elementwise(
        asm="""
            {
              .reg .b32 vec1, vec2, vec_sum;
              .reg .b16 h_low, h_high, h_final;
              .reg .b32 x1, x2;

              mov.b32 vec1, $1;
              mov.b32 vec2, $2;
              mov.b32 x1, $3;
              mov.b32 x2, $4;

              mul.f16x2 vec1, vec1, x1;
              mul.f16x2 vec2, vec2, x2;
              add.f16x2 vec_sum, vec1, vec2;
              mov.b32 {h_high, h_low}, vec_sum; 
              add.f16 h_final, h_high, h_low;              
              cvt.u16.u16 $0,  h_final;
               
            }
        """,
        constraints=(
            "=f,r,r,r,r"
        ),
        args=[x1, x2, vec1, vec2],  # 参数 1 2 3 4
        dtype=(tl.float16), #参数 0
        is_pure=False,
        pack=1,
    )

    return y


@triton.jit
def load_v4_b32(ptr):
    return tl.inline_asm_elementwise(
        asm="ld.global.v4.u32 {$0,$1,$2,$3}, [$4];",
        constraints=("=r,=r,=r,=r,l"),
        args=[ptr],
        dtype=(tl.int32, tl.int32, tl.int32, tl.int32),
        is_pure=False,
        pack=1
    )




@triton.jit
def test_dequant_kernel(
    A_ptr, x_ptr, y_ptr,
    m, k, int4_k,
    stride_am, 
    BLOCK_SIZE : tl.constexpr = 256,
    evict : tl.constexpr = 'evict_last'
):



    row_id = tl.program_id(0)
    acc = 0
    acc = acc.to(tl.float16)


    offs_k =  tl.arange(0, BLOCK_SIZE)
    A_offset = row_id * stride_am + (offs_k)
    
    A_ptr = A_ptr + A_offset
    x_ptr = x_ptr + (offs_k * 2)
    # x_ptr_int = tl.cast(x_ptr, (tl.uint32), bitcast = True)
    for kk in range(0, tl.cdiv(int4_k, BLOCK_SIZE)):
      mask = offs_k < int4_k
      x1, x2, x3, x4 = load_v4_b32(x_ptr)
      a = tl.load(A_ptr,  eviction_policy = evict, mask = mask)
      a1, a2 = dequanti(a)  
      all1 = sum_4_half(a1, a2, x1, x2)     
      a = a >> 8
      a5, a6 = dequanti(a)      
      all2 = sum_4_half(a5, a6, x3, x4) 
      
      acc += tl.sum(all1 + all2, axis=0) 
      offs_k +=  BLOCK_SIZE 
      A_ptr += (BLOCK_SIZE) 
      x_ptr += (BLOCK_SIZE * 2)

    tl.store(y_ptr + row_id, acc)


def basic_ast_analysis():
    """基础 AST 分析"""
    # 获取内核源代码
    
    source = inspect.getsource(gemv_int4_kernel.fn)
    # source = inspect.getsource(test_dequant_kernel.fn)
    print("源代码:")
    print(source)
    print("\n" + "="*50 + "\n")
    
    # 解析为 AST
    tree = ast.parse(source)
    
    # 打印 AST 结构
    print("AST 结构:")
    print(ast.dump(tree, indent=5))
    
    return tree

tree = basic_ast_analysis()

import graphviz
from graphviz import Digraph

def create_ast_graph(tree, max_depth=10):
    """创建 AST 可视化图形"""
    dot = graphviz.Digraph(
        format='png',
        graph_attr={'rankdir': 'TB'},
        node_attr={'shape': 'box', 'style': 'rounded'}
    )
    dot.attr(rankdir='TB')
    
    node_id = 0
    

    global cnt
    cnt = 0

    
    def add_node(parent_id, node, depth=0, later_color_be_defined=False, input_color = None):
        nonlocal node_id
        global cnt
        if depth > max_depth:
            return
        
        current_id = str(node_id)
        node_id += 1
        
        # 确定节点标签
        if isinstance(node, ast.FunctionDef):
            label = f'Function: {node.name}'
            color = 'white'
        elif isinstance(node, ast.For):
            label = 'For'
            color = 'white'
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                label = f'Call: {node.func.attr}'
            else:
                label = 'Call'
            color = 'orange'
        elif isinstance(node, ast.Assign):
            label = 'Assignment'
            color = 'yellow'

            cnt += 1
            print(cnt)
            if cnt == 9:

                later_color_be_defined = True
                input_color = "lightblue"
                
            if cnt == 11:

                later_color_be_defined = True
                input_color = "lightyellow"

            if cnt == 12:

                later_color_be_defined = True
                input_color = "lightpink"
            if cnt == 10:

                later_color_be_defined = True
                input_color = "lightcyan"
            

        elif isinstance(node, ast.Name):
            label = f'Name: {node.id}'
            color = 'lightgrey'

        elif isinstance(node, ast.Constant):
            label = f"Constant \n{repr(node.value)}"
            color = 'white'
        else:
            # print(node)
            label = type(node).__name__
            color = 'white'
        

        if depth == 3 and not isinstance(node, ast.Assign):
            
            later_color_be_defined = True
            input_color = "lemonchiffon"

        if later_color_be_defined is True:
            color = input_color
        if depth == 2 and not isinstance(node, ast.For):
            return 
        # dot.node(current_id, label, style='filled', color=color)
        dot.node(
            current_id,
            label=label,
            shape='Mrecord',
            style='filled',
            fillcolor=color
        )
        if parent_id is not None:
            dot.edge(parent_id, current_id)
        
        # 递归添加子节点
        for child in ast.iter_child_nodes(node):

            if depth == 2 and not isinstance(node, ast.For):
                continue
            add_node(current_id, child, depth + 1, later_color_be_defined, input_color)
    
    add_node(None, tree, later_color_be_defined = False)
    return dot





ast_graph = create_ast_graph(tree)
ast_graph.render('triton_kernel_ast', format='png', cleanup=True)
print("AST 图形已保存为 triton_kernel_ast.png")