from arithmetic import UncertainNumber
from algebra import graph, pgr, gr, draw, draw_ascii, draw_ast, Algebra

# 1. Khởi tạo số bất định X và hàm f(X) = X*X + X + X
X = UncertainNumber({1, 2,3,4,5,6,7,8,9,10})
f = lambda x: (x * x + x +x ** x ) % 256

# 2. Tính đồ thị: Vẫn dùng cây AST, chưa tính toán (Lazy)
g = pgr(f, X)

print(g[1:100])
#print(g.is_computed)   # False: mới chỉ dựng cây AST, chưa hề sinh các điểm (x, y)
#print(g.ast["type"])   # 'pgr'

# 3. Chỉ thực sự tính toán khi gọi print:
#print(g)               # {(1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4)}_u
#print(g.is_computed)   # True: đã tính toán và cache kết quả

# 4. Vẽ đồ thị đã tính toán ở trên:
# Vẽ bằng matplotlib (có thể truyền save_path để lưu file ảnh)
#draw(g, title="Đồ thị hàm số bất định f(X) = X + X", save_path="graph.png")

# Hoặc gọi trực tiếp từ object:
#g.draw()

# 5. Vẽ dạng ký tự ASCII trực tiếp trên Terminal:
#print(draw_ascii(g))

# 6. Xem cấu trúc cây AST:
#draw_ast(g)

#g.draw()