import math
import itertools
import bisect
from fractions import Fraction
from decimal import Decimal, getcontext, ROUND_HALF_EVEN, InvalidOperation
from typing import Callable, Union, List, Tuple, Set, Any

# Độ chính xác cao cho Decimal (512 chữ số thập phân)
getcontext().prec = 512

# Type alias for numeric types (supporting both Real and Complex numbers)
Numeric = Union[int, float, complex, Fraction, Decimal]


# ==================== HELPER: số học chính xác ====================

def _to_exact(v: Any) -> Any:
    """
    Chuyển float sang Fraction (biểu diễn chính xác) nếu có thể.
    int, Fraction, complex giữ nguyên.
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, Fraction):
        return v
    if isinstance(v, float):
        if math.isfinite(v):
            return Fraction(v).limit_denominator(10**15)
        return v  # inf / nan giữ nguyên float
    if isinstance(v, Decimal):
        try:
            return Fraction(v)
        except Exception:
            return v
    return v


def _to_display(v: Any) -> Any:
    """
    Chuyển Fraction thành int nếu mẫu = 1, hoặc float nếu mẫu đơn giản.
    Dùng cho to_set() để giữ kiểu dữ liệu gọn gàng.
    """
    if isinstance(v, Fraction):
        if v.denominator == 1:
            return v.numerator
        # Nếu denominator nhỏ, trả về Fraction (chính xác)
        return v
    if isinstance(v, float):
        if math.isfinite(v) and v == int(v):
            return int(v)
        return v
    if isinstance(v, complex):
        r = _to_display(v.real)
        i = _to_display(v.imag)
        if i == 0:
            return r
        return complex(r, i)
    return v


def _safe_round_val(val: Any) -> Any:
    """Làm sạch giá trị sau tính toán, ưu tiên giữ int hoặc Fraction."""
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return val
    if isinstance(val, Fraction):
        return _to_display(val)
    if isinstance(val, float):
        if not math.isfinite(val):
            return val
        # Chuyển sang Fraction rồi quyết định
        frac = Fraction(val).limit_denominator(10**12)
        if abs(float(frac) - val) < 1e-10 * (1 + abs(val)):
            return _to_display(frac)
        return round(val, 10)
    if isinstance(val, complex):
        r = _safe_round_val(val.real)
        i = _safe_round_val(val.imag)
        if i == 0:
            return r
        return complex(float(r), float(i))
    return val


# ==================== LAGRANGE INTERPOLATION ====================

def lagrange_interpolation(x_nodes: List[int], y_values: List[Numeric]) -> Callable[[Numeric], Numeric]:
    """
    Constructs a generative function f(x) via Lagrange Polynomial Interpolation.
    Sử dụng Fraction để đảm bảo độ chính xác tuyệt đối với số nguyên lớn.
    """
    n = len(x_nodes)
    # Precompute denominators dùng Fraction
    x_frac = [Fraction(xi) for xi in x_nodes]
    y_frac = [_to_exact(yi) for yi in y_values]

    def generative_fn(x: Numeric) -> Numeric:
        x_f = _to_exact(x[0] if isinstance(x, tuple) else x)
        total = Fraction(0)
        for i in range(n):
            xi, yi = x_frac[i], y_frac[i]
            li = Fraction(1)
            for j in range(n):
                if i != j:
                    xj = x_frac[j]
                    li *= (x_f - xj) / (xi - xj)
            if isinstance(yi, (int, Fraction)):
                total += Fraction(yi) * li
            else:
                # complex or float fallback
                return sum(
                    y_values[ii] * float(
                        Fraction.__mul__(
                            Fraction(1),
                            Fraction(1)
                        )
                    )
                    for ii in range(n)
                )
        return _to_display(total)

    return generative_fn


def lagrange_formula_str(x_nodes: List[int], y_values: List[Numeric], var: str = "x") -> str:
    """
    Returns a human-readable symbolic string for the Lagrange interpolating polynomial.
    Dùng Fraction cho hệ số chính xác.
    """
    n = len(x_nodes)
    x_frac = [Fraction(xi) for xi in x_nodes]

    def poly_mul_frac(p: List[Fraction], q: List[Fraction]) -> List[Fraction]:
        result = [Fraction(0)] * (len(p) + len(q) - 1)
        for i, pi in enumerate(p):
            for j, qj in enumerate(q):
                result[i + j] += pi * qj
        return result

    coeffs = [Fraction(0)] * n

    for i in range(n):
        xi, yi = x_frac[i], _to_exact(y_values[i])
        li_poly = [Fraction(1)]
        denom = Fraction(1)
        for j in range(n):
            if i != j:
                xj = x_frac[j]
                li_poly = poly_mul_frac(li_poly, [-xj, Fraction(1)])
                denom *= (xi - xj)
        if isinstance(yi, (int, Fraction)):
            scale = Fraction(yi) / denom
        else:
            scale = Fraction(float(yi)) / denom
        for k in range(len(li_poly)):
            if k < n:
                coeffs[k] += li_poly[k] * scale

    def _fmt(c: Fraction):
        if c.denominator == 1:
            return c.numerator
        # Trả về Fraction nếu đơn giản, float nếu không
        if abs(c.denominator) <= 10000:
            return c
        return float(c)

    coeffs = [_fmt(c) for c in coeffs]

    def _coeff_str(c, with_var: bool) -> str:
        if isinstance(c, Fraction):
            return f"({c})" if with_var else str(c)
        return str(c)

    terms = []
    use_math_style = (var == "x")

    for k in range(n - 1, -1, -1):
        c = coeffs[k]
        if c == 0:
            continue
        if k == 0:
            terms.append(_coeff_str(c, False))
        elif k == 1:
            if c == 1:
                terms.append(var)
            elif c == -1:
                terms.append(f"-{var}")
            else:
                if use_math_style:
                    terms.append(f"{_coeff_str(c, True)}{var}")
                else:
                    terms.append(f"{_coeff_str(c, True)}*{var}")
        else:
            pow_sym = "^" if use_math_style else "**"
            if c == 1:
                terms.append(f"{var}{pow_sym}{k}")
            elif c == -1:
                terms.append(f"-{var}{pow_sym}{k}")
            else:
                if use_math_style:
                    terms.append(f"{_coeff_str(c, True)}{var}^{k}")
                else:
                    terms.append(f"{_coeff_str(c, True)}*{var}**{k}")

    if not terms:
        return "0"

    result = terms[0]
    for t in terms[1:]:
        if t.startswith("-"):
            result += f" - {t[1:]}"
        else:
            result += f" + {t}"
    return result


# ==================== APPROX EQ (chịu được số nguyên siêu lớn) ====================

def _approx_eq(a: Any, b: Any, rel_tol: float = 1e-7, abs_tol: float = 1e-9) -> bool:
    """Checks approximate equality. An toàn với số nguyên siêu lớn và Fraction."""
    if a is None or b is None:
        return False
    # Exact equality (handles int, Fraction, etc.)
    try:
        if a == b:
            return True
    except Exception:
        pass

    # Complex check
    if isinstance(a, complex) or isinstance(b, complex):
        try:
            ca, cb = complex(a), complex(b)
            return (
                math.isclose(ca.real, cb.real, rel_tol=rel_tol, abs_tol=abs_tol)
                and math.isclose(ca.imag, cb.imag, rel_tol=rel_tol, abs_tol=abs_tol)
            )
        except (OverflowError, ValueError):
            return False

    # Fraction / int exact comparison
    try:
        fa = Fraction(a) if not isinstance(a, Fraction) else a
        fb = Fraction(b) if not isinstance(b, Fraction) else b
        diff = abs(fa - fb)
        if diff == 0:
            return True
        scale = max(abs(fa), abs(fb))
        if scale == 0:
            return diff <= Fraction(abs_tol)
        return diff <= Fraction(rel_tol) * scale + Fraction(abs_tol)
    except (TypeError, ValueError, OverflowError, InvalidOperation):
        pass

    try:
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
    except (OverflowError, ValueError):
        return False


# ==================== EXTENDED GCD & DIOPHANTINE ====================

def _extgcd(a: int, b: int) -> Tuple[int, int, int]:
    """Extended Euclidean Algorithm returning (x, y, gcd) such that a*x + b*y = gcd."""
    if b == 0:
        return 1, 0, a
    x1, y1, g = _extgcd(b, a % b)
    return y1, x1 - (a // b) * y1, g


def _solve_diophantine_ranges(
    s_a: int, step_a: int, n_a: int,
    s_b: int, step_b: int, n_b: int,
    op: str, target: int
) -> bool:
    """
    Solves linear Diophantine membership for two range arithmetic sequences in O(log(min(step_a, step_b))).
    """
    if op == "+":
        rem = target - s_a - s_b
        sb = step_b
    elif op == "-":
        rem = target - s_a + s_b
        sb = -step_b
    else:
        return False

    a = step_a
    b = sb

    if b < 0:
        b_abs = -b
        x1, y1, g = _extgcd(a, b_abs)
        if rem % g != 0:
            return False
        scale = rem // g
        a_p = a // g
        b_p = b_abs // g
        u0 = x1 * scale
        v0 = -y1 * scale
        k_min = max(-(-(-u0) // b_p), -(-(-v0) // a_p))
        k_max = min((n_a - 1 - u0) // b_p, (n_b - 1 - v0) // a_p)
        return k_min <= k_max
    else:
        x1, y1, g = _extgcd(a, b)
        if rem % g != 0:
            return False
        scale = rem // g
        a_p = a // g
        b_p = b // g
        u0 = x1 * scale
        v0 = y1 * scale
        k_min = max(-(-(-u0) // b_p), -(-((v0 - (n_b - 1))) // a_p))
        k_max = min((n_a - 1 - u0) // b_p, v0 // a_p)
        return k_min <= k_max


def _get_leaf_linear_params(unc: "UncertainNumber") -> Union[Tuple[Numeric, Numeric, int], None]:
    """Extracts (start, step, count) if the leaf represents an exact linear arithmetic progression."""
    if unc.ast.get("type") != "leaf":
        return None
    if isinstance(getattr(unc, "elements", None), range):
        r = unc.elements
        return (r.start, r.step, len(r))
    if hasattr(unc, "elements") and unc.elements is not None:
        elems = unc.elements
        n = len(elems)
        if n == 0:
            return None
        if n == 1:
            return (elems[0], 0, 1)
        if all(isinstance(x, (int, float, Fraction)) for x in elems):
            e0 = _to_exact(elems[0])
            e_last = _to_exact(elems[-1])
            step = (e_last - e0) / (n - 1) if isinstance(e0, Fraction) else (e_last - e0) / (n - 1)
            # Dùng Fraction để so sánh chính xác
            try:
                step_f = Fraction(e_last - e0, n - 1) if isinstance(e0, (int, Fraction)) else step
                is_arith = all(
                    _to_exact(elems[i]) == e0 + i * step_f
                    for i in range(n)
                )
                if is_arith:
                    return (elems[0], step if isinstance(step, (int, float)) else step_f, n)
            except Exception:
                pass
    if unc.d == (1,):
        v = unc.evaluate_at_index((1,))
        if isinstance(v, (int, float, Fraction)):
            return (v, 0, 1)
    return None


def _get_node_bounds(unc: "UncertainNumber") -> Tuple[Union[float, None], Union[float, None]]:
    """Calculates and caches the [min_val, max_val] interval bounding for the AST node."""
    if hasattr(unc, "_cached_bounds"):
        return unc._cached_bounds

    ast_type = unc.ast.get("type")
    if ast_type == "leaf":
        if isinstance(getattr(unc, "elements", None), range):
            r = unc.elements
            if len(r) == 0:
                bounds = (None, None)
            else:
                s, e = r[0], r[-1]
                bounds = (min(s, e), max(s, e))
        elif hasattr(unc, "elements") and unc.elements:
            real_elems = []
            for e in unc.elements:
                if isinstance(e, complex):
                    real_elems.append(e.real)
                elif isinstance(e, (int, float, Fraction)):
                    real_elems.append(e)
            if real_elems:
                bounds = (min(real_elems), max(real_elems))
            else:
                bounds = (None, None)
        elif unc.d == (1,):
            v = unc.evaluate_at_index((1,))
            val = v.real if isinstance(v, complex) else v
            bounds = (val, val)
        else:
            bounds = (None, None)

    elif ast_type == "op":
        left = unc.ast["left"]
        right = unc.ast["right"]
        op_sym = unc.ast.get("operator_symbol", "+")
        b_l = _get_node_bounds(left)
        b_r = _get_node_bounds(right)
        if b_l[0] is None or b_r[0] is None:
            bounds = (None, None)
        else:
            min_l, max_l = b_l
            min_r, max_r = b_r
            if op_sym == "+":
                bounds = (min_l + min_r, max_l + max_r)
            elif op_sym == "-":
                bounds = (min_l - max_r, max_l - min_r)
            elif op_sym == "*":
                prods = [min_l * min_r, min_l * max_r, max_l * min_r, max_l * max_r]
                bounds = (min(prods), max(prods))
            elif op_sym == "/":
                try:
                    if min_r <= 0 <= max_r:
                        bounds = (None, None)
                    else:
                        divs = [min_l / min_r, min_l / max_r, max_l / min_r, max_l / max_r]
                        bounds = (min(divs), max(divs))
                except Exception:
                    bounds = (None, None)
            else:
                bounds = (None, None)
    else:
        bounds = (None, None)

    unc._cached_bounds = bounds
    return bounds


def _solve_ast_membership(
    unc: "UncertainNumber",
    target: Any,
    rel_tol: float = 1e-7,
    abs_tol: float = 1e-9,
    memo: Union[Set, None] = None,
    depth: int = 0,
) -> bool:
    """
    Core AST Equation Solver engine.
    An toàn với số nguyên siêu lớn và Fraction.
    """
    if depth > 1000:
        return False
    if memo is None:
        memo = set()

    # Memoization key
    if isinstance(target, bool):
        sig_key = target
    elif isinstance(target, int):
        sig_key = target
    elif isinstance(target, Fraction):
        sig_key = target  # hashable
    elif isinstance(target, float):
        sig_key = round(target, 7)
    elif isinstance(target, complex):
        sig_key = (round(target.real, 7), round(target.imag, 7))
    else:
        sig_key = target
    key = (id(unc), sig_key)
    if key in memo:
        return False
    memo.add(key)

    # 1. Interval bounding branch pruning
    b = _get_node_bounds(unc)
    if b[0] is not None and b[1] is not None and isinstance(target, (int, float, Fraction)):
        try:
            t_val = target
            lo, hi = b[0], b[1]
            # So sánh an toàn
            if t_val < lo - abs_tol - rel_tol * abs(lo):
                return False
            if t_val > hi + abs_tol + rel_tol * abs(hi):
                return False
        except (TypeError, OverflowError):
            pass

    ast_type = unc.ast.get("type")

    # 2. Leaf Node
    if ast_type == "leaf":
        if isinstance(getattr(unc, "elements", None), range):
            r = unc.elements
            if len(r) == 0:
                return False
            if isinstance(target, int):
                return target in r
            if target in r:
                return True
            try:
                diff = target - r.start
                k_int = round(diff / r.step)
                if 0 <= k_int < len(r):
                    val = r.start + k_int * r.step
                    return _approx_eq(val, target, rel_tol, abs_tol)
            except Exception:
                pass
            return False

        if hasattr(unc, "elements") and unc.elements is not None:
            elems = unc.elements
            if isinstance(elems, set):
                if target in elems:
                    return True
                return any(_approx_eq(e, target, rel_tol, abs_tol) for e in elems)
            elif isinstance(elems, (list, tuple)):
                if len(elems) == 0:
                    return False
                if (all(isinstance(x, (int, Fraction)) for x in elems)
                        and isinstance(target, (int, Fraction))):
                    # Exact integer/Fraction membership
                    return any(e == target for e in elems)
                if (all(isinstance(x, (int, float)) for x in elems)
                        and isinstance(target, (int, float))):
                    pos = bisect.bisect_left(elems, target - abs_tol)
                    if pos < len(elems) and _approx_eq(elems[pos], target, rel_tol, abs_tol):
                        return True
                    if pos > 0 and _approx_eq(elems[pos - 1], target, rel_tol, abs_tol):
                        return True
                    return False
                return any(_approx_eq(e, target, rel_tol, abs_tol) for e in elems)

        if unc.d == (1,):
            v = unc.evaluate_at_index((1,))
            return _approx_eq(v, target, rel_tol, abs_tol)

        if len(unc.d) == 1:
            n = unc.d[0]
            if n <= 100:
                return any(_approx_eq(unc.f(i), target, rel_tol, abs_tol) for i in range(1, n + 1))

        return False

    # 3. Pointwise Operation Node
    elif unc.ast.get("space_type") == "pointwise":
        n = unc.d[0] if unc.d else 1
        try:
            import sympy
            formula_str = unc.get_formula(["x"])
            x_sym = sympy.Symbol("x")
            expr = sympy.sympify(formula_str.replace("^", "**"))
            sols = sympy.solve(sympy.Eq(expr, target), x_sym)
            for sol in sols:
                try:
                    val_c = complex(sol.evalf())
                    if abs(val_c.imag) < 1e-9:
                        r = round(val_c.real)
                        if 1 <= r <= n and abs(val_c.real - r) < 1e-4:
                            if _approx_eq(unc.evaluate_at_index((r,)), target, rel_tol, abs_tol):
                                return True
                except Exception:
                    pass
        except Exception:
            pass

        if n <= 10000:
            return any(_approx_eq(unc.evaluate_at_index((i,)), target, rel_tol, abs_tol) for i in range(1, n + 1))
        return False

    # 4. Minkowski Operation Node
    elif ast_type == "op":
        left: "UncertainNumber" = unc.ast["left"]
        right: "UncertainNumber" = unc.ast["right"]
        op_sym = unc.ast.get("operator_symbol", "+")

        # Fast-path 4a: Linear Diophantine
        lin_l = _get_leaf_linear_params(left)
        lin_r = _get_leaf_linear_params(right)
        if lin_l and lin_r and op_sym in ("+", "-"):
            sl, stepl, nl = lin_l
            sr, stepr, nr = lin_r
            if (
                isinstance(sl, int) and isinstance(stepl, int)
                and isinstance(sr, int) and isinstance(stepr, int)
                and isinstance(target, int)
            ):
                if stepl > 0 and stepr > 0:
                    return _solve_diophantine_ranges(sl, stepl, nl, sr, stepr, nr, op_sym, target)

        # Fast-path 4b: Invert operation
        len_l = left.index_length if left.d else 0
        len_r = right.index_length if right.d else 0

        def _get_cands(node: "UncertainNumber"):
            if node.ast.get("type") == "leaf" and hasattr(node, "elements") and node.elements is not None:
                if len(node.elements) <= 1000:
                    return node.elements
            if node.index_length <= 100:
                try:
                    return node.to_set()
                except Exception:
                    return None
            return None

        cands_r = _get_cands(right)
        cands_l = _get_cands(left)

        branch_right = True
        if cands_r is not None and cands_l is not None:
            branch_right = len(cands_r) <= len(cands_l)
        elif cands_r is not None:
            branch_right = True
        elif cands_l is not None:
            branch_right = False
        else:
            branch_right = len_r <= len_l

        if branch_right and cands_r is not None:
            for vr in cands_r:
                if op_sym == "+":
                    req_l = target - vr
                    if _solve_ast_membership(left, req_l, rel_tol, abs_tol, memo, depth + 1):
                        return True
                elif op_sym == "-":
                    req_l = target + vr
                    if _solve_ast_membership(left, req_l, rel_tol, abs_tol, memo, depth + 1):
                        return True
                elif op_sym == "*":
                    if _approx_eq(vr, 0, rel_tol, abs_tol):
                        if _approx_eq(target, 0, rel_tol, abs_tol) and left.index_length > 0:
                            return True
                    else:
                        try:
                            req_l = _to_exact(target) / _to_exact(vr)
                            if _solve_ast_membership(left, req_l, rel_tol, abs_tol, memo, depth + 1):
                                return True
                        except Exception:
                            pass
                elif op_sym == "/":
                    if not _approx_eq(vr, 0, rel_tol, abs_tol):
                        try:
                            req_l = _to_exact(target) * _to_exact(vr)
                            if _solve_ast_membership(left, req_l, rel_tol, abs_tol, memo, depth + 1):
                                return True
                        except Exception:
                            pass
                elif op_sym == "**":
                    if _approx_eq(vr, 0, rel_tol, abs_tol):
                        if _approx_eq(target, 1, rel_tol, abs_tol) and left.index_length > 0:
                            return True
                    else:
                        try:
                            # Dùng Fraction nếu vr là int/Fraction
                            if isinstance(vr, (int, Fraction)) and isinstance(target, int) and target > 0:
                                # req_l = target^(1/vr) — kiểm tra chính xác
                                vr_f = Fraction(vr)
                                if vr_f.denominator == 1 and vr_f.numerator > 0:
                                    p = vr_f.numerator
                                    # Kiểm tra xem target có phải p-th power không
                                    guess = round(target ** (1.0 / p))
                                    for candidate in [guess - 1, guess, guess + 1]:
                                        if candidate > 0 and candidate ** p == target:
                                            if _solve_ast_membership(left, candidate, rel_tol, abs_tol, memo, depth + 1):
                                                return True
                            req_l = target ** (1.0 / vr)
                            if _solve_ast_membership(left, req_l, rel_tol, abs_tol, memo, depth + 1):
                                return True
                        except Exception:
                            pass

        elif not branch_right and cands_l is not None:
            for vl in cands_l:
                if op_sym == "+":
                    req_r = target - vl
                    if _solve_ast_membership(right, req_r, rel_tol, abs_tol, memo, depth + 1):
                        return True
                elif op_sym == "-":
                    req_r = vl - target
                    if _solve_ast_membership(right, req_r, rel_tol, abs_tol, memo, depth + 1):
                        return True
                elif op_sym == "*":
                    if _approx_eq(vl, 0, rel_tol, abs_tol):
                        if _approx_eq(target, 0, rel_tol, abs_tol) and right.index_length > 0:
                            return True
                    else:
                        try:
                            req_r = _to_exact(target) / _to_exact(vl)
                            if _solve_ast_membership(right, req_r, rel_tol, abs_tol, memo, depth + 1):
                                return True
                        except Exception:
                            pass
                elif op_sym == "/":
                    if _approx_eq(target, 0, rel_tol, abs_tol):
                        if _approx_eq(vl, 0, rel_tol, abs_tol):
                            if any(not _approx_eq(vr2, 0, rel_tol, abs_tol) for vr2 in cands_r or [1]):
                                return True
                    else:
                        try:
                            req_r = _to_exact(vl) / _to_exact(target)
                            if not _approx_eq(req_r, 0, rel_tol, abs_tol):
                                if _solve_ast_membership(right, req_r, rel_tol, abs_tol, memo, depth + 1):
                                    return True
                        except Exception:
                            pass
                elif op_sym == "**":
                    if isinstance(vl, (int, Fraction)) and vl > 0 and not _approx_eq(vl, 1, rel_tol, abs_tol) and isinstance(target, int) and target > 0:
                        try:
                            req_r = Fraction(math.log(target), math.log(float(vl)))
                            if _solve_ast_membership(right, req_r, rel_tol, abs_tol, memo, depth + 1):
                                return True
                        except Exception:
                            pass

        return False

    return False


class UncertainNumber:
    """
    Represents an Uncertain Number U(K) defined strictly via its Canonical Form (f_X, d_X).
    Hỗ trợ số nguyên siêu lớn và độ chính xác cao qua Fraction / Python native int.
    """

    def __init__(
        self,
        data: Union[List[Numeric], Set[Numeric], None] = None,
        generative_fn: Union[Callable, None] = None,
        index_domain: Union[Tuple[int, ...], None] = None,
        ast_node: Union[dict, None] = None,
    ):
        if data is not None:
            if isinstance(data, range):
                n = len(data)
                self.d: Tuple[int, ...] = (n,)
                self.elements = data
                if n == 0:
                    self.f = lambda x: None
                    self._formula_template = lambda v: "0"
                elif n == 1:
                    val = data[0]
                    self.f = lambda x: val
                    if val == 1:
                        self._formula_template = lambda var: f"{var}"
                    elif val == -1:
                        self._formula_template = lambda var: f"-{var}"
                    else:
                        self._formula_template = lambda var, c=val: f"{c}*{var}"
                else:
                    start = data[0]
                    step = data.step
                    self.f = lambda x: start + ((x[0] if isinstance(x, tuple) else x) - 1) * step
                    if step == 1 and start == 1:
                        self._formula_template = lambda v: f"{v}"
                    elif step == 1 and start == 0:
                        self._formula_template = lambda v: f"{v} - 1"
                    elif step == 1:
                        self._formula_template = lambda v: f"{start - 1} + {v}" if start > 1 else f"{v} - {1 - start}"
                    else:
                        self._formula_template = lambda v: f"{start} + ({v} - 1) * {step}"
                self.ast: dict = {"type": "leaf"}

            else:
                data_list = list(data)
                try:
                    y_values = sorted(data_list, key=lambda x: (
                        isinstance(x, complex),
                        float(x.real) if isinstance(x, complex) else float(x) if isinstance(x, (int, float, Fraction)) else 0
                    ))
                except Exception:
                    y_values = data_list
                n = len(y_values)

                self.d: Tuple[int, ...] = (n,)
                self.elements = y_values

                is_numeric = all(
                    isinstance(v, (int, float, complex, Fraction)) and not isinstance(v, bool)
                    for v in y_values
                )
                if is_numeric and n > 0:
                    if n == 1:
                        val = y_values[0]
                        self.f = lambda x: val
                        if val == 1:
                            self._formula_template = lambda var: f"{var}"
                        elif val == -1:
                            self._formula_template = lambda var: f"-{var}"
                        else:
                            self._formula_template = lambda var, c=val: f"{c}*{var}"
                    else:
                        # Kiểm tra cấp số cộng — dùng Fraction để chính xác
                        all_int_or_frac = all(isinstance(v, (int, Fraction)) for v in y_values)
                        if all_int_or_frac:
                            e0 = Fraction(y_values[0])
                            e_last = Fraction(y_values[-1])
                            step_f = Fraction(e_last - e0, n - 1) if n > 1 else Fraction(0)
                            is_arithmetic = all(
                                Fraction(y_values[i]) == e0 + i * step_f
                                for i in range(n)
                            )
                        else:
                            step = (float(y_values[-1]) - float(y_values[0])) / (n - 1)
                            is_arithmetic = False
                            if n <= 100:
                                is_arithmetic = all(
                                    abs(float(y_values[i]) - (float(y_values[0]) + i * step)) < 1e-9
                                    for i in range(n)
                                )
                            else:
                                is_arithmetic = (
                                    abs(float(y_values[1]) - (float(y_values[0]) + step)) < 1e-9 and
                                    abs(float(y_values[n // 2]) - (float(y_values[0]) + (n // 2) * step)) < 1e-9 and
                                    abs(float(y_values[-1]) - (float(y_values[0]) + (n - 1) * step)) < 1e-9
                                )
                            step_f = step

                        if is_arithmetic:
                            y0 = y_values[0]
                            _step_f = step_f  # capture

                            def _arith_f(x, _y0=y0, _step=_step_f):
                                idx = x[0] if isinstance(x, tuple) else x
                                result = _y0 + (idx - 1) * _step
                                return _to_display(result) if isinstance(result, Fraction) else result

                            self.f = _arith_f

                            # Formula template
                            if all_int_or_frac:
                                s = int(step_f) if step_f.denominator == 1 else step_f
                                y0d = int(e0) if e0.denominator == 1 else e0
                            else:
                                s = step_f
                                y0d = y_values[0]

                            if s == 1 and y0d == 1:
                                self._formula_template = lambda v: f"{v}"
                            elif s == 1 and y0d == 0:
                                self._formula_template = lambda v: f"{v} - 1"
                            elif s == 1:
                                self._formula_template = (
                                    (lambda v: f"{y0d - 1} + {v}") if y0d > 1
                                    else (lambda v: f"{v} - {1 - y0d}")
                                )
                            else:
                                self._formula_template = lambda v, _y=y0d, _s=s: f"{_y} + ({v} - 1) * {_s}"
                        elif n <= 10:
                            x_nodes = list(range(1, n + 1))
                            self.f = lagrange_interpolation(x_nodes, y_values)
                            _xn, _yv = x_nodes[:], y_values[:]
                            self._formula_template = lambda v, xn=_xn, yv=_yv: lagrange_formula_str(xn, yv, v)
                        else:
                            # Tra cứu mảng O(1)
                            _yv_cap = y_values[:]

                            def _fast_eval(x: Numeric, _yv=_yv_cap, _n=n) -> Numeric:
                                idx_val = x[0] if isinstance(x, tuple) else x
                                if isinstance(idx_val, (int, float, Fraction)) and 1 <= idx_val <= _n:
                                    if isinstance(idx_val, int) or (isinstance(idx_val, float) and idx_val.is_integer()):
                                        return _yv[int(idx_val) - 1]
                                    if isinstance(idx_val, Fraction) and idx_val.denominator == 1:
                                        return _yv[int(idx_val) - 1]
                                    x_int = int(idx_val)
                                    if x_int == _n:
                                        return _yv[-1]
                                    frac_part = idx_val - x_int
                                    return _yv[x_int - 1] * (1 - frac_part) + _yv[x_int] * frac_part
                                return None

                            self.f = _fast_eval
                            self._formula_template = lambda v: f"PiecewiseLinear({v})"
                else:
                    self.f = lambda idx: self.elements[int(idx[0] if isinstance(idx, tuple) else idx) - 1] if self.elements and 1 <= int(idx[0] if isinstance(idx, tuple) else idx) <= len(self.elements) else None
                    self._formula_template = lambda v: f"DiscreteLookup({v})"
                self.ast: dict = {"type": "leaf"}

        elif generative_fn is not None or index_domain is not None or ast_node is not None:
            self.d = tuple(index_domain) if index_domain is not None else ()
            self.ast = ast_node if ast_node is not None else {"type": "custom_fn"}
            self.f = generative_fn if generative_fn is not None else (lambda idx: self.evaluate_at_index(idx))
            self._formula_template = ast_node.get("formula_template") if ast_node else None

        else:
            raise ValueError(
                "Must provide either raw numeric data OR a generative function 'f' with index domain tuple 'd'."
            )

    def evaluate_at_index(self, index_key: Any) -> Any:
        """Evaluates the generative function f_X at a specific index key or index tuple."""
        if not isinstance(index_key, tuple):
            index_key = (index_key,)

        if self.ast.get("type") in ("leaf", "custom_fn"):
            idx = index_key[0] if len(index_key) == 1 else index_key
            return self.f(idx)

        elif self.ast.get("type") == "op":
            left: "UncertainNumber" = self.ast["left"]
            right: "UncertainNumber" = self.ast["right"]
            operator_fn: Callable = self.ast["operator_fn"]
            space_type = self.ast.get("space_type", "minkowski")

            if space_type == "pointwise":
                if left.d == right.d:
                    idx_left = index_key
                    idx_right = index_key
                elif left.d == (1,):
                    idx_left = (1,)
                    idx_right = index_key
                elif right.d == (1,):
                    idx_left = index_key
                    idx_right = (1,)
                else:
                    idx_left = index_key
                    idx_right = index_key
            else:  # minkowski
                left_dim = len(left.d)
                idx_left = index_key[:left_dim]
                idx_right = index_key[left_dim:]

            val_left = left.evaluate_at_index(idx_left)
            val_right = right.evaluate_at_index(idx_right)
            return operator_fn(val_left, val_right)

        return self.f(index_key[0] if len(index_key) == 1 else index_key)

    def to_set(self) -> Set[Any]:
        """
        Lazy Evaluation: sinh tập kết quả từ miền chỉ số.
        Dùng so sánh chính xác cho int/Fraction, không làm tròn float tùy tiện.
        """
        if self.d == (0,) or (self.ast.get("type") == "leaf" and hasattr(self, "elements") and not self.elements):
            return set()

        index_ranges = [range(1, n + 1) for n in self.d]

        results = set()
        for idx in itertools.product(*index_ranges):
            try:
                val = self.evaluate_at_index(idx)
                if val is None:
                    continue
                val = _safe_round_val(val)
                results.add(val)
            except ZeroDivisionError:
                continue
        return results

    # ==================== ABSTRACT AST COMPOSITION OPERATOR ====================

    def compose(self, other: Any, operator_fn: Union[Callable, None] = None) -> "UncertainNumber":
        """Abstract composition: Joins two AST trees under a custom operator function."""
        if not isinstance(other, UncertainNumber):
            other = UncertainNumber([other])

        if operator_fn is None:
            operator_fn = lambda a, b: a * b

        new_d = self.d + other.d

        ast_node = {
            "type": "op",
            "left": self,
            "right": other,
            "operator_fn": operator_fn,
            "operator_symbol": "*",
            "space_type": "minkowski",
        }

        res = UncertainNumber(
            generative_fn=lambda idx: res.evaluate_at_index(idx),
            index_domain=new_d,
            ast_node=ast_node,
        )
        return res

    def to_set_key(self) -> tuple:
        """Returns a hashable tuple representation of its set values."""
        try:
            s = self.to_set()
            formatted_elems = []
            for elem in s:
                if isinstance(elem, UncertainNumber):
                    formatted_elems.append(elem.to_set_key())
                else:
                    formatted_elems.append(elem)
            return tuple(sorted(formatted_elems, key=lambda x: str(x)))
        except Exception:
            return (id(self),)

    def __hash__(self):
        return hash(self.to_set_key())

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, UncertainNumber):
            return self.to_set_key() == other.to_set_key()
        return False

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, UncertainNumber):
            return str(self) < str(other)
        return False

    def flat_index_to_tuple(self, flat_idx: int, zero_based: bool = True) -> Tuple[int, ...]:
        """Converts a 1D linear integer index into a multi-dimensional scenario index tuple."""
        if not self.d or self.d == (0,):
            return ()

        total_size = math.prod(self.d)
        k0 = flat_idx if zero_based else (flat_idx - 1)

        if k0 < 0:
            k0 += total_size

        if not (0 <= k0 < total_size):
            raise IndexError(f"Index {flat_idx} out of range for UncertainNumber with {total_size} scenarios (domain {self.d}).")

        tuple_idx = []
        for dim_size in reversed(self.d):
            if dim_size <= 0:
                tuple_idx.append(1)
            else:
                tuple_idx.append((k0 % dim_size) + 1)
                k0 //= dim_size

        return tuple(reversed(tuple_idx))

    def tuple_to_flat_index(self, tuple_idx: Tuple[int, ...], zero_based: bool = True) -> int:
        """Converts a multi-dimensional index tuple into a 1D linear integer index."""
        if len(tuple_idx) != len(self.d):
            raise ValueError(f"Dimension mismatch: expected {len(self.d)} elements, got {len(tuple_idx)}")

        k0 = 0
        for idx_val, dim_size in zip(tuple_idx, self.d):
            if not (1 <= idx_val <= dim_size):
                raise IndexError(f"Index component {idx_val} out of bounds for dimension size {dim_size}")
            k0 = k0 * dim_size + (idx_val - 1)

        return k0 if zero_based else (k0 + 1)

    def __len__(self) -> int:
        return math.prod(self.d) if self.d and self.d != (0,) else 0

    def __iter__(self):
        return iter(sorted(list(self.to_set()), key=lambda x: (isinstance(x, complex), str(x))))

    def __getitem__(self, index: Any):
        if isinstance(index, tuple):
            return self.evaluate_at_index(index)
        elif isinstance(index, int):
            tuple_idx = self.flat_index_to_tuple(index, zero_based=True)
            val = self.evaluate_at_index(tuple_idx)
            return _safe_round_val(val)
        elif isinstance(index, slice):
            total_size = len(self)
            start, stop, step = index.indices(total_size)
            return [self[i] for i in range(start, stop, step)]
        else:
            raise TypeError(f"UncertainNumber indices must be integers or tuples, not {type(index).__name__}")

    def get_formula(self, var_names: Union[List[str], Tuple[str, ...], None] = None) -> str:
        """Returns the symbolic mathematical formula f(x_1, x_2, ...) of the generative function."""
        num_vars = len(self.d) if self.d and self.d != (0,) else 1
        if var_names is None:
            if num_vars == 1 and self.ast.get("type") == "leaf":
                var_names = ["x"]
            elif num_vars == 1:
                var_names = ["x_1"]
            else:
                var_names = [f"x_{i + 1}" for i in range(num_vars)]

        if self.ast.get("type") in ("leaf", "custom_fn"):
            if hasattr(self, "_formula_template") and self._formula_template is not None:
                v = var_names[0] if var_names else "x"
                return self._formula_template(v)
            elif hasattr(self, "elements") and len(self.elements) == 1:
                val = self.elements[0]
                v = var_names[0] if var_names else "x"
                return (f"{val}*{v}" if val not in (1, -1) else (f"{v}" if val == 1 else f"-{v}"))
            else:
                v_str = ", ".join(var_names) if var_names else "x"
                return f"f({v_str})"

        elif self.ast.get("type") == "op":
            left: "UncertainNumber" = self.ast["left"]
            right: "UncertainNumber" = self.ast["right"]
            op_sym = self.ast.get("operator_symbol", "+")
            space_type = self.ast.get("space_type", "minkowski")

            if space_type == "pointwise":
                left_vars = var_names
                right_vars = var_names
            else:
                left_dim = len(left.d) if left.d and left.d != (0,) else 1
                left_vars = var_names[:left_dim]
                right_vars = var_names[left_dim:]

            left_str = left.get_formula(left_vars)
            right_str = right.get_formula(right_vars)
            if op_sym == "*":
                return f"({left_str})*({right_str})"
            return f"({left_str}) {op_sym} ({right_str})"

        v_str = ", ".join(var_names) if var_names else "x"
        return f"f({v_str})"

    @property
    def formula(self) -> str:
        """Returns the canonical form formula representation."""
        num_vars = len(self.d) if self.d and self.d != (0,) else 1
        if num_vars == 1 and self.ast.get("type") == "leaf":
            var_names = ["x"]
            head = "f(x)"
        elif num_vars == 1:
            var_names = ["x_1"]
            head = "f(x_1)"
        else:
            var_names = [f"x_{i + 1}" for i in range(num_vars)]
            head = f"f({', '.join(var_names)})"
        return f"{head} = {self.get_formula(var_names)}"

    def __mul__(self, other: Any) -> "UncertainNumber":
        """Overloads '*' to act as an abstract AST composition operator."""
        return self.compose(other)

    def __repr__(self):
        try:
            s = self.to_set()
            if not s:
                return "{}_u"
            sorted_items = sorted(
                list(s),
                key=lambda x: (
                    1 if isinstance(x, UncertainNumber) else 0,
                    x.to_set_key() if isinstance(x, UncertainNumber) else str(x),
                ),
            )
            inner = ", ".join(repr(x) if isinstance(x, UncertainNumber) else str(x) for x in sorted_items)
            return f"{{{inner}}}_u"
        except Exception:
            return f"UncertainNumber(d={self.d})"

    def print_model(self):
        print(f"d={self.d}\nf={self.formula}")

    @property
    def index_length(self):
        """Calculates the total number of scenarios."""
        return math.prod(self.d)

    def contains(self, target: Any, rel_tol: float = 1e-7, abs_tol: float = 1e-9) -> bool:
        """
        Kiểm tra một phần tử 'target' có thuộc số bất định hay không.
        An toàn với số nguyên siêu lớn.
        """
        return _solve_ast_membership(self, target, rel_tol=rel_tol, abs_tol=abs_tol)

    def __contains__(self, item: Any) -> bool:
        return self.contains(item)

    def solve_equation(
        self,
        target: Any = 0,
        var_names: Union[List[str], Tuple[str, ...], None] = None,
    ) -> List[Tuple[int, ...]]:
        """Giải phương trình f(x_1, x_2, ..., x_k) = target trên cây AST."""
        num_vars = len(self.d) if self.d and self.d != (0,) else 1
        if var_names is None:
            if num_vars == 1 and self.ast.get("type") == "leaf":
                var_names = ["x"]
            elif num_vars == 1:
                var_names = ["x_1"]
            else:
                var_names = [f"x_{i + 1}" for i in range(num_vars)]

        formula_body = self.get_formula(var_names)
        try:
            import sympy
            symbols = [sympy.Symbol(v) for v in var_names]
            expr = sympy.sympify(formula_body.replace("^", "**"))
            eq = sympy.Eq(expr, target)

            if len(symbols) == 1:
                sols = sympy.solve(eq, symbols[0])
                valid_tuples = []
                n_max = self.d[0] if self.d else 1
                for s in sols:
                    try:
                        c_val = complex(s.evalf())
                        if abs(c_val.imag) < 1e-9:
                            r = round(c_val.real)
                            if 1 <= r <= n_max and abs(c_val.real - r) < 1e-4:
                                if _approx_eq(self.evaluate_at_index((r,)), target):
                                    valid_tuples.append((r,))
                    except Exception:
                        pass
                return valid_tuples
        except Exception:
            pass

        if self.index_length <= 100000:
            valid_tuples = []
            index_ranges = [range(1, n + 1) for n in self.d]
            for idx in itertools.product(*index_ranges):
                val = self.evaluate_at_index(idx)
                if _approx_eq(val, target):
                    valid_tuples.append(idx)
            return valid_tuples

        return []

    def __str__(self):
        return self.__repr__()

    # ==================== FUNCTIONAL SPACE OPERATORS ====================

    @staticmethod
    def pw(fn: Callable, *args: Any) -> "UncertainNumber":
        return pw(fn, *args)

    @staticmethod
    def epw(fn: Callable, *args: Any) -> "UncertainNumber":
        return epw(fn, *args)

    @staticmethod
    def m(fn: Callable, *args: Any) -> "UncertainNumber":
        return m(fn, *args)

    @staticmethod
    def em(fn: Callable, *args: Any) -> "UncertainNumber":
        return em(fn, *args)


# ==================== MODULE-LEVEL CONVENIENCE FUNCTIONS ====================

def _to_unc(val: Any) -> UncertainNumber:
    if isinstance(val, UncertainNumber):
        return val
    if isinstance(val, (set, list, tuple)):
        return UncertainNumber(val)
    return UncertainNumber({val})


class _PwExpr:
    """Symbolic tracer for Pointwise expressions."""

    def __init__(
        self,
        eval_fn: Callable[[Any], Any],
        formula_fn: Callable[[str], str],
        unc: "UncertainNumber" = None,
    ):
        self.eval_fn = eval_fn
        self.formula_fn = formula_fn
        self.unc = unc

    def eval(self, idx: Any) -> Any:
        return self.eval_fn(idx)

    def get_formula(self, var: str) -> str:
        return self.formula_fn(var)

    def __add__(self, other):
        return _make_pw_bin("+", self, other, lambda a, b: a + b)

    def __radd__(self, other):
        return _make_pw_bin("+", other, self, lambda a, b: a + b)

    def __sub__(self, other):
        return _make_pw_bin("-", self, other, lambda a, b: a - b)

    def __rsub__(self, other):
        return _make_pw_bin("-", other, self, lambda a, b: a - b)

    def __mul__(self, other):
        return _make_pw_bin("*", self, other, lambda a, b: a * b)

    def __rmul__(self, other):
        return _make_pw_bin("*", other, self, lambda a, b: a * b)

    def __truediv__(self, other):
        return _make_pw_bin("/", self, other, lambda a, b: a / b)

    def __rtruediv__(self, other):
        return _make_pw_bin("/", other, self, lambda a, b: a / b)

    def __floordiv__(self, other):
        return _make_pw_bin("//", self, other, lambda a, b: a // b)

    def __rfloordiv__(self, other):
        return _make_pw_bin("//", other, self, lambda a, b: a // b)

    def __mod__(self, other):
        return _make_pw_bin("%", self, other, lambda a, b: a % b)

    def __rmod__(self, other):
        return _make_pw_bin("%", other, self, lambda a, b: a % b)

    def __pow__(self, other):
        return _make_pw_bin("**", self, other, lambda a, b: a ** b)

    def __rpow__(self, other):
        return _make_pw_bin("**", other, self, lambda a, b: a ** b)

    def __neg__(self):
        return _PwExpr(
            eval_fn=lambda idx: -self.eval_fn(idx),
            formula_fn=lambda var: f"-({self.formula_fn(var)})",
        )

    def __pos__(self):
        return self

    def __abs__(self):
        return _PwExpr(
            eval_fn=lambda idx: abs(self.eval_fn(idx)),
            formula_fn=lambda var: f"abs({self.formula_fn(var)})",
        )


def _make_pw_bin(op_sym: str, left: Any, right: Any, op_fn: Callable) -> _PwExpr:
    if isinstance(left, _PwExpr) and isinstance(right, _PwExpr):
        eval_fn = lambda idx: op_fn(left.eval(idx), right.eval(idx))
        if op_sym == "+":
            formula_fn = lambda v: f"({left.get_formula(v)}) + ({right.get_formula(v)})"
        elif op_sym == "-":
            formula_fn = lambda v: f"({left.get_formula(v)}) - ({right.get_formula(v)})"
        elif op_sym == "*":
            formula_fn = lambda v: f"({left.get_formula(v)})*({right.get_formula(v)})"
        elif op_sym == "**":
            formula_fn = lambda v: f"({left.get_formula(v)})**({right.get_formula(v)})"
        else:
            formula_fn = lambda v: f"({left.get_formula(v)}) {op_sym} ({right.get_formula(v)})"
        return _PwExpr(eval_fn, formula_fn)

    elif isinstance(left, _PwExpr):
        eval_fn = lambda idx: op_fn(left.eval(idx), right)
        if op_sym == "+":
            formula_fn = lambda v: f"{left.get_formula(v)} + {right}"
        elif op_sym == "-":
            formula_fn = lambda v: f"{left.get_formula(v)} - {right}"
        elif op_sym == "*":
            formula_fn = lambda v: f"{left.get_formula(v)}*{right}"
        elif op_sym == "**":
            formula_fn = lambda v: f"({left.get_formula(v)})**{right}"
        else:
            formula_fn = lambda v: f"{left.get_formula(v)} {op_sym} {right}"
        return _PwExpr(eval_fn, formula_fn)

    elif isinstance(right, _PwExpr):
        eval_fn = lambda idx: op_fn(left, right.eval(idx))
        if op_sym == "*":
            formula_fn = lambda v: f"{left}*( {right.get_formula(v)})" if " " in right.get_formula(v) else f"{left}*({right.get_formula(v)})"
        elif op_sym == "+":
            formula_fn = lambda v: f"{left} + ({right.get_formula(v)})"
        elif op_sym == "-":
            formula_fn = lambda v: f"{left} - ({right.get_formula(v)})"
        elif op_sym == "**":
            formula_fn = lambda v: f"{left}**({right.get_formula(v)})"
        else:
            formula_fn = lambda v: f"{left} {op_sym} ({right.get_formula(v)})"
        return _PwExpr(eval_fn, formula_fn)

    else:
        val = op_fn(left, right)
        return _PwExpr(lambda idx: val, lambda v: str(val))


def pw(fn: Callable, *args: Any) -> UncertainNumber:
    """
    Point-wise Space (o)_1 functional operator.
    Definition 3.2: (f)_1(A) := {f(x) : x in A}_u
    """
    if not args:
        return UncertainNumber({fn()})

    unc_args = [_to_unc(a) for a in args]
    d0 = unc_args[0].d

    for u in unc_args:
        if u.d != d0:
            raise ValueError(
                f"Point-wise Space (o)_1 requires identical index domains. Got {u.d} and {d0}."
            )

    try:
        pw_args = [
            _PwExpr(
                eval_fn=(lambda idx, unc=u: unc.evaluate_at_index(idx)),
                formula_fn=(lambda var, unc=u: unc.get_formula([var])),
                unc=u,
            )
            for u in unc_args
        ]
        res_expr = fn(*pw_args)
        if isinstance(res_expr, _PwExpr):
            return UncertainNumber(
                generative_fn=lambda idx: res_expr.eval(idx),
                index_domain=d0,
                ast_node={
                    "type": "custom_fn",
                    "space_type": "pointwise",
                    "formula_template": lambda v: res_expr.get_formula(v),
                },
            )
        elif isinstance(res_expr, UncertainNumber):
            return res_expr
        elif isinstance(res_expr, (int, float, complex, Fraction)):
            return UncertainNumber({res_expr})
    except Exception:
        pass

    generative_fn = lambda idx: fn(*(u.evaluate_at_index(idx) for u in unc_args))
    return UncertainNumber(
        generative_fn=generative_fn,
        index_domain=d0,
        ast_node={"type": "custom_fn", "space_type": "pointwise"},
    )


def epw(fn: Callable, *args: Any) -> UncertainNumber:
    """
    Extended Point-wise Space (o)_1' functional operator.
    Supports broadcasting between scalars and multi-element UncertainNumbers.
    """
    if not args:
        return UncertainNumber({fn()})

    unc_args = [_to_unc(a) for a in args]
    non_scalar_domains = [u.d for u in unc_args if u.d != (1,)]

    if not non_scalar_domains:
        target_d = (1,)
    else:
        target_d = non_scalar_domains[0]
        for d in non_scalar_domains:
            if d != target_d:
                raise ValueError(
                    f"Incompatible multi-element domains for (o)_1': {d} and {target_d}."
                )

    try:
        pw_args = [
            _PwExpr(
                eval_fn=(lambda idx, unc=u: unc.evaluate_at_index((1,) if unc.d == (1,) else idx)),
                formula_fn=(lambda var, unc=u: unc.get_formula([var])),
                unc=u,
            )
            for u in unc_args
        ]
        res_expr = fn(*pw_args)
        if isinstance(res_expr, _PwExpr):
            return UncertainNumber(
                generative_fn=lambda idx: res_expr.eval(idx),
                index_domain=target_d,
                ast_node={
                    "type": "custom_fn",
                    "space_type": "pointwise",
                    "formula_template": lambda v: res_expr.get_formula(v),
                },
            )
        elif isinstance(res_expr, UncertainNumber):
            return res_expr
        elif isinstance(res_expr, (int, float, complex, Fraction)):
            return UncertainNumber({res_expr})
    except Exception:
        pass

    generative_fn = lambda idx: fn(
        *(u.evaluate_at_index((1,) if u.d == (1,) else idx) for u in unc_args)
    )
    return UncertainNumber(
        generative_fn=generative_fn,
        index_domain=target_d,
        ast_node={"type": "custom_fn", "space_type": "pointwise"},
    )


def m(fn: Callable, *args: Any) -> UncertainNumber:
    """
    Minkowski Space (o)_m functional operator.
    """
    if not args:
        return UncertainNumber({fn()})

    unc_args = [_to_unc(a) for a in args]

    try:
        res = fn(*unc_args)
        if isinstance(res, UncertainNumber):
            if any(res is u for u in unc_args):
                pass  # fall through
            else:
                return res
        elif isinstance(res, (int, float, complex, Fraction, set, list, tuple)):
            return _to_unc(res)
    except Exception:
        pass

    new_d = sum((u.d for u in unc_args), ())

    def generative_fn(idx_tuple: Any) -> Numeric:
        if not isinstance(idx_tuple, tuple):
            idx_tuple = (idx_tuple,)
        arg_vals = []
        curr = 0
        for u in unc_args:
            dim = len(u.d)
            sub_idx = idx_tuple[curr: curr + dim]
            curr += dim
            arg_vals.append(u.evaluate_at_index(sub_idx))
        return fn(*arg_vals)

    return UncertainNumber(
        generative_fn=generative_fn,
        index_domain=new_d,
        ast_node={"type": "custom_fn", "space_type": "minkowski"},
    )


def em(fn: Callable, *args: Any) -> UncertainNumber:
    """
    Extended Minkowski Space (o)_m' functional operator.
    """
    if not args:
        return UncertainNumber({fn()})

    unc_args = [_to_unc(a) for a in args]

    if len(unc_args) == 2:
        from .EMinkowskiArithmetic import EMinkowskiArithmetic
        if unc_args[0].d == (1,) or unc_args[1].d == (1,):
            return EMinkowskiArithmetic.em(unc_args[0], unc_args[1], fn)

    if len(unc_args) == 1:
        from .EMinkowskiArithmetic import EMinkowskiArithmetic
        unc_target = unc_args[0]
        probe = 4

        try:
            result_mul = fn(probe)
            if isinstance(result_mul, (int, float, Fraction)) and probe != 0:
                scalar_val = Fraction(result_mul, probe) if isinstance(result_mul, int) else result_mul / probe
                probe2 = 9
                result_mul2 = fn(probe2)
                if isinstance(result_mul2, (int, float, Fraction)):
                    sv2 = Fraction(result_mul2, probe2) if isinstance(result_mul2, int) else result_mul2 / probe2
                    if _approx_eq(sv2, scalar_val):
                        scalar_unc = UncertainNumber({scalar_val})
                        return EMinkowskiArithmetic.em(scalar_unc, unc_target, lambda x, y: x * y)
        except Exception:
            pass

        try:
            result_pow = fn(probe)
            if isinstance(result_pow, (int, float)) and probe > 0 and result_pow > 0:
                import math as _math
                p_val = _math.log(result_pow) / _math.log(probe)
                probe2 = 9
                result_pow2 = fn(probe2)
                if isinstance(result_pow2, (int, float)) and result_pow2 > 0:
                    p_val2 = _math.log(result_pow2) / _math.log(probe2)
                    if abs(p_val2 - p_val) < 1e-9:
                        power_unc = UncertainNumber({p_val})
                        return EMinkowskiArithmetic.em(unc_target, power_unc, lambda x, y: x ** y)
        except Exception:
            pass

    return m(fn, *unc_args)
