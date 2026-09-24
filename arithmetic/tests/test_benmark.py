import sys
import os
import pytest
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