# Arithmetic
Mathematics of Uncertainty: Uncertain number arithmetic and arithmetic spaces.

Trần, M. C. (2026). Extended logic and mathematics of uncertainty. <br/>
https://doi.org/10.5281/zenodo.22031881

## Author & Contact

**Trần Mạnh Cường**  
*Alumnus, Faculty of Mathematics & Informatics (K11)*  
*Thai Nguyen University of Sciences (TNUS), Vietnam*

* **Research Focus:** Mathematics of Uncertainty
* **Email:** tmcuong2408@gmail.com
* **Phone:** (+84) 353-237-140
* **Location:** DJ7 Street, Thoi Hoa, Ho Chi Minh City, Vietnam

## Usage

```python
from arithmetic import UncertainNumber, Arithmetic
from arithmetic import pw, epw, m, em

X = UncertainNumber({1, 2, 3, 4})
print(f"X = {X}")

# 1. Phép trừ trong Không gian Đơn điểm (pw) -> Triệt tiêu rác số học
res_pw = Arithmetic.sub(X, X, space="pw")
print(f"(X - X)_pw = {res_pw}")

# 2. Phép trừ trong Không gian Minkowski (m) -> Phình khoảng độc lập
res_m = Arithmetic.sub(X, X, space="m")
print(f"(X - X)_m  = {res_m}")

# 3. Phép nhân trong Không gian Mở rộng (epw / em)
res_epw = Arithmetic.mul(X, 10, space="epw")
print(f"(X * 10)_epw = {res_epw}")

res_em = Arithmetic.mul(10, X, space="em")
print(f"(10*X)_em  = {res_em}")

# 4. Phép nhân phân số trong Không gian Minkowski Mở rộng (em) -> Giải nghịch ảnh
Y = UncertainNumber({2, 3, 4})
res_em_half = Arithmetic.mul(0.5, Y, space="em")
print(f"(0.5 * {{2, 3, 4}})_em = {res_em_half}")

# 5. Sử dụng hàm pw, epw, m, em với đầu vào lambda:python
f_pw = pw(lambda x: x**2 + 5*x + 6, X)
print(f"pw(lambda x: x^2 + 5x + 6, X) = {f_pw}")
# 6. Quan hệ hai ngôi yếu và lấy giá trị chân lý với hàm mu:
from arithmetic import mu

A = UncertainNumber({1, 2})
B = UncertainNumber({1, 2})

# Gọi qua phương thức của đối tượng:
val = A.mu("<=", B)        # hoặc A.mu(B, "<=")
print(f"A.mu('<=', B) = {val}")  # 0.75

# Hoặc gọi qua hàm cấp module:
val_mod = mu(A, B, "<=")
print(f"mu(A, B, '<=') = {val_mod}")  # 0.75



# Cách khởi tạo số bất định có trọng số

from arithmetic import UncertainNumber

# A: 1 có trọng số 0.2, 2 có trọng số 0.8
A = UncertainNumber({1, 2}, weights={1: 0.2, 2: 0.8})

# B: 1 có trọng số 0.5, 2 có trọng số 0.5
B = UncertainNumber({1, 2}, weights={1: 0.5, 2: 0.5})

# Tính mu(A <= B)
# Cặp (1,1): 1 <= 1 -> 0.2 * 0.5 = 0.1
# Cặp (1,2): 1 <= 2 -> 0.2 * 0.5 = 0.1
# Cặp (2,1): 2 <= 1 -> 0
# Cặp (2,2): 2 <= 2 -> 0.8 * 0.5 = 0.4
# Tổng mu = 0.1 + 0.1 + 0.4 = 0.6
print(A <= B)       # 0.6
print(A.mu("<=", B)) # 0.6
```

## Minkowski Lifting — Kế thừa hàm vô hướng lên không gian Minkowski

Định nghĩa tổng quát **(o)\_m**: với hàm vô hướng `f(a, b, ...)` trên số thực/phức, kế thừa lên không gian Minkowski:

$$f(A, B, \ldots) = \{f(a, b, \ldots) : a \in A,\ b \in B,\ \ldots\}$$

Miền chỉ số mới là tích Descartes $d_A \times d_B \times \cdots$, hàm sinh $f(i, j, \ldots) = f(A[i],\, B[j], \ldots)$.

Sau khi `import arithmetic`, các hàm `max()` và `min()` builtin của Python được **tự động override** để tuân thủ định nghĩa này. Mọi hàm vô hướng khác đều có thể lift thủ công bằng `lift_m`.

```python
from arithmetic import s, lift_m, m
import math

a = s(1, 2, 4, 6, 7, 8, 9, 20, 100)
b = s(1, 2)

# max(A, B) = { max(a, b) : a ∈ A, b ∈ B }
print(max(a, b))   # {2, 4, 6, 7, 8, 9, 20, 100}_u  ✓ (không phải {1, 2}_u sai)

# min(A, B) = { min(a, b) : a ∈ A, b ∈ B }
print(min(a, b))   # {1, 2}_u

# max/min vẫn hoạt động bình thường với số thực:
print(max(3, 7))   # 7
print(min(3, 7))   # 3

# Lift hàm vô hướng bất kỳ với lift_m(f, A, B, ...):
#   lift_m(f, A, B) = { f(a, b) : a ∈ A, b ∈ B }

print(lift_m(math.gcd, a, b))    # { gcd(x, y) : x ∈ a, y ∈ b }
print(lift_m(math.hypot, a, b))  # { hypot(x, y) : x ∈ a, y ∈ b }
print(lift_m(pow, a, b))         # { x^y : x ∈ a, y ∈ b }

# Tương đương — m() cũng là Minkowski lifting tổng quát:
print(m(math.gcd, a, b))         # như lift_m(math.gcd, a, b)
```

| Hàm | Mô tả |
|-----|-------|
| `max(A, B)` | `{max(a,b) : a∈A, b∈B}` — override tự động sau `import arithmetic` |
| `min(A, B)` | `{min(a,b) : a∈A, b∈B}` — override tự động sau `import arithmetic` |
| `lift_m(f, A, B, ...)` | `{f(a,b,...) : a∈A, b∈B, ...}` — lift hàm vô hướng bất kỳ |
| `m(f, A, B, ...)` | Tương đương `lift_m`, hỗ trợ cả lambda và hàm có tên |
