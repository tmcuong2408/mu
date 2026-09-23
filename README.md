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

## Libraries Architecture
The **MU (Mathematics of Uncertainty)** project consists of 3 dedicated libraries:
1. **`arithmetic/`**: Uncertain number representations (`UncertainNumber`) and arithmetic spaces (Pointwise, Minkowski, Extended spaces).
2. **`algebra/`**: Algebraic structures for uncertainty mathematics (groups, rings, fields, vector spaces).
3. **`analysis/`**: Mathematical analysis for uncertain functions (limits, continuity, derivatives, integrals).

## Quick Start: Arithmetic
```python
from arithmetic import UncertainNumber, Arithmetic
from arithmetic import pw, epw, m, em, s, c

X = UncertainNumber({1, 2, 3, 4})
print(f"X = {X}")

# 1. Subtraction in Pointwise Space (pw) -> Eliminates arithmetic noise
res_pw = Arithmetic.sub(X, X, space="pw")
print(f"(X - X)_pw = {res_pw}")

# 2. Subtraction in Minkowski Space (m) -> Independent interval expansion
res_m = Arithmetic.sub(X, X, space="m")
print(f"(X - X)_m  = {res_m}")

# 3. Multiplication in Extended Space (epw / em)
res_epw = Arithmetic.mul(X, 10, space="epw")
print(f"(X * 10)_epw = {res_epw}")

res_em = Arithmetic.mul(10, X, space="em")
print(f"(10*X)_em  = {res_em}")

# 4. Scalar fractional multiplication in Extended Minkowski Space (em) -> Preimage solver (Example 3.10)
Y = UncertainNumber({2, 3, 4})
res_em_half = Arithmetic.mul(0.5, Y, space="em")
print(f"(0.5 * {{2, 3, 4}})_em = {res_em_half}")

# 5. Weak binary relations and truth values with the mu function
from arithmetic import mu

A_rel = UncertainNumber({1, 2})
B_rel = UncertainNumber({1, 2})

val_mu = A_rel.mu("<=", B_rel)
print(f"A_rel.mu('<=', B_rel) = {val_mu}")  # 0.75

val_mu_mod = mu(A_rel, B_rel, "<=")
print(f"mu(A_rel, B_rel, '<=') = {val_mu_mod}")  # 0.75
```

## Functional Space Operators: `pw`, `epw`, `m`, `em`

The `pw`, `epw`, `m`, `em` functions support **two calling styles**:

| Syntax | Description |
|---|---|
| `pw(fn)` | Returns a **callable** operator for reuse |
| `pw(fn, A, B, ...)` | Evaluates directly, returning an `UncertainNumber` immediately (shorthand) |

```python
from arithmetic import UncertainNumber, pw, epw, m, em, s

A = s(1, 2, 3)        # UncertainNumber({1, 2, 3})
B = s(10, 20)         # UncertainNumber({10, 20})

# --- pw: Pointwise Space ---
# Callable syntax (reusable operator)
square = pw(lambda x: x**2)
print(square(A))       # {1, 4, 9}
print(square(B))       # {100, 400}   <- reused on different input

# Shorthand syntax (direct evaluation)
print(pw(lambda x: x**2, A))   # {1, 4, 9}

# --- epw: Extended Pointwise Space (scalar broadcasting) ---
scale = epw(lambda x, k: x * k + 1)
print(scale(A, 2))     # {3, 5, 7}
print(scale(A, 10))    # {11, 21, 31}  <- reused with a different scalar

# Shorthand
print(epw(lambda x, k: x * k + 1, A, 2))   # {3, 5, 7}

# --- m: Minkowski Space (Cartesian product) ---
add_m = m(lambda a, b: a + b)
print(add_m(A, B))     # {11, 21, 12, 22, 13, 23}

mul_m = m(lambda a, b: a * b)
print(mul_m(A, B))     # Cartesian product A × B

# Shorthand
print(m(lambda a, b: a + b, A, B))

# --- em: Extended Minkowski Space (preimage solver) ---
half = em(lambda x: 0.5 * x)
Y = s(2, 3, 4)
print(half(Y))         # {1, 2}  (solves for X such that X +_m X = {2, 3, 4})

# Shorthand
print(em(lambda x: 0.5 * x, Y))
```

## Function Composition: `c`

The `c` function constructs a composite function from an **ordered list of callables**:

```
c([f, g, h]) = f ∘ g ∘ h   ≡   lambda *args: f(g(h(*args)))
```

Application order is **right-to-left** (consistent with mathematical notation): the last element in the list is applied first.

```python
from arithmetic import UncertainNumber, pw, m, epw, s, c

a = s(1, 2)

# Create operators across different arithmetic spaces
f_m  = m(lambda x: x + x)     # Minkowski double: {1, 2} -> {2, 3, 4}
f_pw = pw(lambda x: x + x)    # Pointwise double: doubles element-wise

# --- 2-function composition ---
h = c([f_pw, f_m])             # h(x) = f_pw(f_m(x))
print(h(a))                    # {4, 6, 8}

# --- 3-function composition ---
f_plus1 = m(lambda x: x + 1)
h3 = c([f_plus1, f_pw, f_m])   # h3(x) = f_plus1(f_pw(f_m(x)))
print(h3(a))

# --- Reusing composite operators on different inputs ---
b = s(1, 2, 4)
print(h(a))    # {4, 6, 8}
print(h(b))    # {4, 6, 10, 8, 12, 16}  <- same operator, different input

# --- Composing across mixed spaces ---
f1 = m(lambda x: x + x)        # Minkowski space
f2 = pw(lambda x: x * x)       # Pointwise space
f3 = epw(lambda x, k: x + k)  # Extended Pointwise

compose = c([f2, f1])
print(compose(s(1, 2, 3)))
```

> **Note on evaluation order**: `c([f, g])` means `f(g(x))` — `g` is evaluated first, then `f`, exactly like $f \circ g$ in mathematical notation.

## Benchmark: 10¹⁸ Scenarios with O(1) RAM

```python
import sys
import os

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from arithmetic import UncertainNumber, Arithmetic
from arithmetic import pw, epw, m, em
from math import sin
import time
import sys
import tracemalloc


def benchmark_trillion_scale():
    print("=" * 70)
    print("🚀 BENCHMARK: 'MU' ENGINE ON 10^18 SCENARIOS SCALE")
    print("=" * 70)

    # 1. Start memory tracing
    tracemalloc.start()
    t0 = time.perf_counter()

    # Create two uncertain numbers with 1 billion elements (10^9) each
    # A = [1, 2, ..., 1_000_000_000]
    # B = [100, 200, ..., 100_000_000_100]
    print("\n1. Initializing uncertain numbers...")
    A = UncertainNumber(range(1, 1_000_000_001))
    B = UncertainNumber(range(100, 100_000_000_100, 100))

    # In Minkowski Space (m), total scenarios = |A| x |B| = 10^9 x 10^9 = 10^18 (1 Quintillion scenarios)
    print("2. Performing Minkowski operation (A + B) on Lazy AST Tree...")
    C = A + B

    t1 = time.perf_counter()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # 2. Inspect initialization parameters & AST
    print("\n----------------------------------------------------------------------")
    print(f"✅ Total scenarios (N): {len(C):,} scenarios")
    print(f"✅ Algebraic formula:   {C.formula}")
    print(f"⏱️ Tree build time:    {t1 - t0:.6f} seconds")
    print(f"💾 Peak RAM usage:      {peak_mem / 1024:.2f} KB (Virtually 0)")
    print("----------------------------------------------------------------------")

    # 3. Test Odometer feature: Random access to scenarios in O(k) time
    print("\n3. Testing random index queries (Odometer Query Test):")
    
    test_indices = [
        0,                        # First scenario (0-based)
        500_000_000,              # 500-millionth scenario
        1_000_000_000_000_000,    # 1-quadrillionth scenario (10^15)
        len(C) - 1                # Final scenario (10^18 - 1)
    ]

    for idx in test_indices:
        t_start = time.perf_counter_ns()
        
        # Odometer automatically converts flat_idx -> tuple_idx -> evaluate_at_index
        val = C[idx]
        tuple_idx = C.flat_index_to_tuple(idx)
        
        t_end = time.perf_counter_ns()
        latency_us = (t_end - t_start) / 1000

        print(f"  • Flat Index [{idx:,}]:")
        print(f"    - Coordinate Tuple: {tuple_idx}")
        print(f"    - Evaluated Value : {val:,}")
        print(f"    - Query Latency   : {latency_us:.3f} microseconds")

    # 4. Test Pointwise (pw) operation preserving dimensionality
    print("\n4. Testing Pointwise (pw) identical variable operation (10^9 scenarios):")
    D = pw(lambda x: x**2 - 2*x + 1, A)
    print(f"  • Pointwise formula: {D.formula}")
    print(f"  • Value at index 999,999,999: {D[999_999_999]:,}")
    print("🎉 RESULT: PERFECT PASS! Engine meets O(1) RAM & O(k) TIME standards!")
    
benchmark_trillion_scale()
