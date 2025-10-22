
# each_warp_reduce_compute = 3
# blockdim = 8


# t = [i for i in range(0, 64)]

# def is_subset_using_set(arr1, arr2):
#     """使用集合验证 (最快)"""
#     set2 = set(arr2)
#     return all(num in set2 for num in arr1)

# def is_subset_using_list(arr1, arr2):
#     """使用列表验证 (较慢)"""
#     for num in arr1:
#         if num not in arr2:
#             return False
#     return True


# data = []
# # 确认循环的数据 m 完全覆盖了 0 ~ 4095
# for bx in range(0, 32):
#     for ty in range(0, 8):
#          m = ( ((blockdim * bx + ty) // each_warp_reduce_compute) ) 
#          data.append(m)

# print(data)
# print(is_subset_using_list(t, data))


text = """
warp = 0, sum[topx] =0.5795  tid = 0
warp = 0, sum[topx] =0.7909  tid = 0
warp = 0, sum[topx] =4.5301  tid = 0
warp = 0, sum[topx] =-0.6082  tid = 0
warp = 0, sum[topx] =7.1258  tid = 0
warp = 0, sum[topx] =-0.6082  tid = 0
warp = 0, sum[topx] =4.5301  tid = 0
warp = 0, sum[topx] =-15.7976  tid = 0"""

text = text.replace("warp = 0, ","")
text = text.replace("sum[topx] ","")
text = text.replace("tid","")
text = text.replace(",","")
text = text.replace(" ","")
text = text.replace("\n","")
text = text.replace("=",",")
text = text.split(",")
s = 0
for i in range(1,  len(text), 2):
    s =  s + float(text[i])
print(s)