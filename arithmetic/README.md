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
