# MU
Mathematics of Uncertrainty

<div align="center">
  <h1>Abstract</h1>
</div>

<p>
  In modern computational science, engineering, and decision-making, handling uncertain data remains a critical challenge[cite: 1]. Traditional approaches, such as classical <b>Interval Arithmetic</b>, suffer severely from the <i>Dependency Problem</i>—where repeated algebraic variables generate artificial "arithmetic noise" (spurious solutions) and cause explosive interval bounds[cite: 1]. Similarly, Fuzzy Logic and Probabilistic methods often demand strict prior distributions and incur high computational overheads when scenario space expands[cite: 1].
</p>

<p>
  This repository presents <b><code>MU</code> (Mathematics of Uncertainty)</b>, an open-source Python library implementing a novel foundational mathematical framework designed to eliminate interval explosion and resolve uncertain data processing natively[cite: 1].
</p>

<h3>Key Architectural & Theoretical Features</h3>

<ul>
  <li>
    <b>Canonical Form & Lazy Evaluation Engine:</b> Represents uncertain numbers via generative functions on index domains <i>(f<sub>X</sub>, d<sub>X</sub>)</i>[cite: 1]. By maintaining algebraic correlation histories, <code>MU</code> eliminates spurious arithmetic noise[cite: 1]. Leveraging a <b>Lazy Evaluation</b> design, it stores and manipulates infinite-dimensional scenario sets with minimal memory overhead, evaluating exact values only upon explicit query[cite: 1].
  </li>
  <li>
    <b>Four-Space Arithmetic Architecture:</b>
    <ul>
      <li><i>Point-wise Spaces (&circ;)<sub>1</sub> & (&circ;)<sub>1'</sub>:</i> Preserve internal variable identity and algebraic consistency (e.g., <i>(X &minus; X)<sub>1</sub> = {0}<sub>u</sub></i>), guaranteeing exact balance in structural operations[cite: 1].</li>
      <li><i>Minkowski Spaces (&circ;)<sub>m</sub> & (&circ;)<sub>m'</sub>:</i> Model independent scenario interactions, noise propagation, and inverse arithmetic operators[cite: 1].</li>
    </ul>
  </li>
  <li>
    <b>Structural Algebra & Index-Domain Solvers:</b> Transforms complex set-theoretic operations (intersections, membership queries) and weak inequality relations <i>&mu;(A &mathcal;R; B)</i> into exact index-domain root-finding problems[cite: 1], drastically reducing time complexity[cite: 1].
  </li>
  <li>
    <b>Uncertain Calculus & Differential Forms:</b> Implements a fully consistent calculus engine encompassing limits, continuity, difference quotients, derivatives, and exact differential-form path integrals over <i>k</i>-dimensional parameter spaces <i>[0, 1]<sup>k</sup></i>[cite: 1].
  </li>
</ul>

<p>
  <b><code>MU</code></b> provides a rigorous, vectorized, and scalable computational engine tailored for noisy data processing, uncertain system modeling, inverse problems, and robust decision-making[cite: 1].
</p>

<hr />

<p>
  <b>Keywords:</b> <i>Mathematics of Uncertainty, Interval Arithmetic, Canonical Form, Lazy Evaluation, Scenario Index Domain, Weak Binary Relation, Uncertain Calculus, Python Library.</i>[cite: 1]
</p>
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
