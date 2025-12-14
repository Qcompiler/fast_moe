from pathlib import Path
import triton

tmp = Path("/home/chenyidong/newstart/bandwidth/qmoe/microkernel")


filename = "save_global_auto_gen.ptx"
temp_file = (tmp / filename)


kernel = triton.compile(str(temp_file))