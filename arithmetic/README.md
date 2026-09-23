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

# 4. Scalar fractional multiplication in Extended Minkowski Space (em) -> Preimage solver
Y = UncertainNumber({2, 3, 4})
res_em_half = Arithmetic.mul(0.5, Y, space="em")
print(f"(0.5 * {{2, 3, 4}})_em = {res_em_half}")

# 5. Using pw, epw, m, em with lambda functions
f_pw = pw(lambda x: x**2 + 5*x + 6, X)
print(f"pw(lambda x: x^2 + 5x + 6, X) = {f_pw}")

# 6. Weak binary relations and truth values with the mu function:
from arithmetic import mu

A = UncertainNumber({1, 2})
B = UncertainNumber({1, 2})

# Call via object method:
val = A.mu("<=", B)        # or A.mu(B, "<=")
print(f"A.mu('<=', B) = {val}")  # 0.75

# Or call via module-level function:
val_mod = mu(A, B, "<=")
print(f"mu(A, B, '<=') = {val_mod}")  # 0.75



# Initializing weighted uncertain numbers:

from arithmetic import UncertainNumber

# A: 1 has weight 0.2, 2 has weight 0.8
A = UncertainNumber({1, 2}, weights={1: 0.2, 2: 0.8})

# B: 1 has weight 0.5, 2 has weight 0.5
B = UncertainNumber({1, 2}, weights={1: 0.5, 2: 0.5})

# Calculate mu(A <= B)
# Pair (1,1): 1 <= 1 -> 0.2 * 0.5 = 0.1
# Pair (1,2): 1 <= 2 -> 0.2 * 0.5 = 0.1
# Pair (2,1): 2 <= 1 -> 0
# Pair (2,2): 2 <= 2 -> 0.8 * 0.5 = 0.4
# Total mu = 0.1 + 0.1 + 0.4 = 0.6
print(A <= B)       # 0.6
print(A.mu("<=", B)) # 0.6
```

## Minkowski Lifting — Extending Scalar Functions to Minkowski Space

General definition **(o)\_m**: for any scalar function `f(a, b, ...)` defined on real/complex numbers, extend it to Minkowski space:

$$f(A, B, \ldots) = \{f(a, b, \ldots) : a \in A,\ b \in B,\ \ldots\}$$

The new coordinate domain is the Cartesian product $d_A \times d_B \times \cdots$, with generator $f(i, j, \ldots) = f(A[i],\, B[j], \ldots)$.

Upon `import arithmetic`, Python's builtin `max()` and `min()` functions are **automatically overridden** to adhere to this definition. Any other scalar function can be lifted manually using `lift_m`.

```python
from arithmetic import s, lift_m, m
import math

a = s(1, 2, 4, 6, 7, 8, 9, 20, 100)
b = s(1, 2)

# max(A, B) = { max(a, b) : a ∈ A, b ∈ B }
print(max(a, b))   # {2, 4, 6, 7, 8, 9, 20, 100}_u

# min(A, B) = { min(a, b) : a ∈ A, b ∈ B }
print(min(a, b))   # {1, 2}_u

# max/min continue to function normally with standard real numbers:
print(max(3, 7))   # 7
print(min(3, 7))   # 3

# Lift any scalar function with lift_m(f, A, B, ...):
#   lift_m(f, A, B) = { f(a, b) : a ∈ A, b ∈ B }

print(lift_m(math.gcd, a, b))    # { gcd(x, y) : x ∈ a, y ∈ b }
print(lift_m(math.hypot, a, b))  # { hypot(x, y) : x ∈ a, y ∈ b }
print(lift_m(pow, a, b))         # { x^y : x ∈ a, y ∈ b }

# Equivalent — m() also provides general Minkowski lifting:
print(m(math.gcd, a, b))         # same as lift_m(math.gcd, a, b)
```

| Function | Description |
|---|---|
| `max(A, B)` | `{max(a, b) : a ∈ A, b ∈ B}` — automatically overridden upon `import arithmetic` |
| `min(A, B)` | `{min(a, b) : a ∈ A, b ∈ B}` — automatically overridden upon `import arithmetic` |
| `lift_m(f, A, B, ...)` | `{f(a, b, ...) : a ∈ A, b ∈ B, ...}` — lifts any scalar function |
| `m(f, A, B, ...)` | Equivalent to `lift_m`, supporting both lambdas and named functions |
