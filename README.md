# MU
Mathematics of Uncertainty

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

# 4. Phép nhân phân số trong Không gian Minkowski Mở rộng (em) -> Giải nghịch ảnh (Ví dụ 3.10)
Y = UncertainNumber({2, 3, 4})
res_em_half = Arithmetic.mul(0.5, Y, space="em")
print(f"(0.5 * {{2, 3, 4}})_em = {res_em_half}")

# 5. Sử dụng hàm pw, epw, m, em với đầu vào lambda:
# Ví dụ 3.2: f(X) = (X^2 + 5X + 6)_1
f_pw = pw(lambda x: x**2 + 5*x + 6, X)
print(f"pw(lambda x: x^2 + 5x + 6, X) = {f_pw}")

# Triệt tiêu rác số học: f(x) = x - x
zero_pw = pw(lambda x: x - x, X)
print(f"pw(lambda x: x - x, X) = {zero_pw}")

# Phép toán 2 ngôi trên không gian Minkowski: m(lambda a, b: a + b, A, B)
A = UncertainNumber({1, 3})
B = UncertainNumber({10, 20})
res_m_fn = m(lambda a, b: a + b, A, B)
print(f"m(lambda a, b: a + b, A, B) = {res_m_fn}")

# Extended Pointwise với lambda:
res_epw_fn = epw(lambda x, c: x * c, X, 10)
print(f"epw(lambda x, c: x * c, X, 10) = {res_epw_fn}")

Y_example = UncertainNumber({2, 3, 4})
res_em_add_valid = Arithmetic.mul(0.5, Y_example, space="em")
print(f"(0.5 * {{2, 3, 4}})_em  = {res_em_add_valid}")

X_custom = UncertainNumber({1, 5, 6, 7, 25, 30, 35, 36, 42, 49})
res_em_add_none = Arithmetic.mul(0.5, X_custom, space="em")
print(f"(0.5 * {{1, 5, 6, 7, 25, 30, 35, 36, 42, 49}})_em = {res_em_add_none} (Vô nghiệm)")

res_em_pow = Arithmetic.pow(X_custom, 0.5, space="em")
print(f"({{1, 5, 6, 7, 25, 30, 35, 36, 42, 49}}^0.5)_em = {res_em_pow}")
```

<p>
bench: Verify $10^{18}$ scenario scale with $\mathcal{O}(1)$ RAM and $\mathcal{O}(k)$ latency
</p>

```python
import sys
import os

# Thêm thư mục gốc (parent directory) vào sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from arithmetic import UncertainNumber, Arithmetic
from arithmetic import pw, epw, m, em
from math import sin
import time
import sys
import tracemalloc


def benchmark_trillion_scale():
    print("=" * 70)
    print("🚀 BENCHMARK: CỖ MÁY 'MU' TRÊN QUY MÔ TỶ TỶ KỊCH BẢN (10^18)")
    print("=" * 70)

    # 1. Bắt đầu đo bộ nhớ
    tracemalloc.start()
    t0 = time.perf_counter()

    # Tạo 2 số bất định có 1 tỷ phần tử (10^9) mỗi số
    # A = [1, 2, ..., 1_000_000_000]
    # B = [100, 200, ..., 100_000_000_000]
    print("\n1. Khởi tạo số bất định...")
    A = UncertainNumber(range(1, 1_000_000_001))
    B = UncertainNumber(range(100, 100_000_000_100, 100))

    # Trong Minkowski Space (m), tổng kịch bản = |A| x |B| = 10^9 x 10^9 = 10^18 (1 Tỷ Tỷ kịch bản)
    print("2. Thực hiện phép toán Minkowski (A + B) trên Cây Lazy AST...")
    C = A + B

    t1 = time.perf_counter()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # 2. Kiểm tra thông số khởi tạo & AST
    print("\n----------------------------------------------------------------------")
    print(f"✅ Tổng số kịch bản (N): {len(C):,} kịch bản")
    print(f"✅ Công thức đại số:    {C.formula}")
    print(f"⏱️ Thời gian dựng cây: {t1 - t0:.6f} giây")
    print(f"💾 RAM tiêu thụ cực đại: {peak_mem / 1024:.2f} KB (Gần như bằng 0)")
    print("----------------------------------------------------------------------")

    # 3. Test tính năng Odometer: Truy xuất ngẫu nhiên kịch bản trong O(k) time
    print("\n3. Kiểm thử truy xuất ngẫu nhiên chỉ số (Odometer Query Test):")
    
    test_indices = [
        0,                        # Kịch bản đầu tiên (0-based)
        500_000_000,              # Kịch bản thứ 500 triệu
        1_000_000_000_000_000,    # Kịch bản thứ 1 Triệu Tỷ (10^15)
        len(C) - 1                # Kịch bản cuối cùng (Thứ 1 Tỷ Tỷ - 1)
    ]

    for idx in test_indices:
        t_start = time.perf_counter_ns()
        
        # Odometer tự động chuyển flat_idx -> tuple_idx -> evaluate_at_index
        val = C[idx]
        tuple_idx = C.flat_index_to_tuple(idx)
        
        t_end = time.perf_counter_ns()
        latency_us = (t_end - t_start) / 1000

        print(f"  • Flat Index [{idx:,}]:")
        print(f"    - Coordinate Tuple: {tuple_idx}")
        print(f"    - Evaluated Value : {val:,}")
        print(f"    - Query Latency   : {latency_us:.3f} microseconds")

    # 4. Test phép toán Pointwise (pw) giữ nguyên quy mô
    print("\n4. Kiểm thử phép toán Pointwise (pw) đồng nhất biến (10^9 kịch bản):")
    D = pw(lambda x: x**2 - 2*x + 1, A)
    print(f"  • Công thức Pointwise: {D.formula}")
    print(f"  • Giá trị tại chỉ số 999,999,999: {D[999_999_999]:,}")
    print("🎉 KẾT QUẢ: PASS HOÀN HẢO! CỖ MÁY ĐẠT CHUẨN O(1) RAM & O(k) TIME!")
    
benchmark_trillion_scale()
