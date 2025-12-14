
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
      q_shift = ((offs_k % elements_per_sample) * W_nbits).to(tl.int32)
      a = ((a >> q_shift) & unpack_mask) - 8 
      a = a.to(tl.float16)

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
    
    
import ast
import graphviz

def visualize_ast(tree, filename="ast_tree"):
    """Simple function to visualize AST"""

    
    # Create graph
    dot = graphviz.Digraph(
        format='png',
        graph_attr={'rankdir': 'TB'},
        node_attr={'shape': 'box', 'style': 'rounded'}
    )
    
    max_depth = [0]  # 使用列表以便在递归中修改
    
    def add_nodes(ast_node, current_depth, parent=None):
        """递归添加节点并统计深度"""
        # 更新最大深度
        if current_depth > max_depth[0]:
            max_depth[0] = current_depth
        
        # 创建节点ID和标签
        node_id = f"node_{id(ast_node)}_{current_depth}"
        
        # 在标签中添加深度信息
        # label = f"{ast_node.__class__.__name__}\n(depth: {current_depth})"
        # node_id = str(id(ast_node))
        
        # Create label
        label = ast_node.__class__.__name__
        if isinstance(ast_node, ast.Constant):
            label = f"Constant\n{repr(ast_node.value)}"
        elif isinstance(ast_node, ast.Name):
            label = f"Name\n'{ast_node.id}'"
        elif isinstance(ast_node, ast.FunctionDef):
            label = f"Function\n'{ast_node.name}'"
        elif isinstance(ast_node, ast.BinOp):
            label = f"BinOp\n{ast_node.op.__class__.__name__}"
        
        dot.node(node_id, label)
        
        # Connect to parent
        if parent:
            dot.edge(parent, node_id)
        
        # 递归处理子节点
        for child in ast.iter_child_nodes(ast_node):

            if current_depth == 2 and not isinstance(ast_node, ast.For):
                continue
            add_nodes(child, current_depth + 1, node_id)
    
    # 构建图形
    add_nodes(tree, 0)
    # Build the graph
    print(f"AST最大深度: {max_depth[0]}")
    

    # Save and render
    dot.render(filename, format = "png", cleanup=False)
    print(f"AST saved as '{filename}.png'")







def basic_ast_analysis():
    """基础 AST 分析"""
    # 获取内核源代码
    
    source = inspect.getsource(gemv_int4_kernel.fn)
    # source = inspect.getsource(test_dequant_kernel.fn)
    # print("源代码:")
    # print(source)
    print("\n" + "="*50 + "\n")
    
    # 解析为 AST
    tree = ast.parse(source)
    return tree
    
if __name__ == "__main__":
    # Test with simple code
    simple_code = """
    x = 10 + 5
    y = x * 2
    """
    # Parse the code
    # tree = ast.parse(simple_code)
    # visualize_ast(tree, "simple_ast")

    tree = basic_ast_analysis()
    visualize_ast(tree, "simple_ast")