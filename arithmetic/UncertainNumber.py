import math
import itertools
import bisect
from typing import Callable, Union, List, Tuple, Set, Any

# Type alias for numeric types (supporting both Real and Complex numbers)
Numeric = Union[int, float, complex]



def lagrange_interpolation(x_nodes: List[int], y_values: List[Numeric]) -> Callable[[Numeric], Numeric]:
    """
    Constructs a generative function f(x) via Lagrange Polynomial Interpolation[cite: 1].
    Maps index nodes {1, ..., n} to corresponding numeric target values[cite: 1].
    """
    n = len(x_nodes)

    def generative_fn(x: Numeric) -> Numeric:
        total = 0
        for i in range(n):
            xi, yi = x_nodes[i], y_values[i]
            li = 1
            for j in range(n):
                if i != j:
                    xj = x_nodes[j]
                    li *= (x - xj) / (xi - xj)
            total += yi * li
        return total

    return generative_fn


def lagrange_formula_str(x_nodes: List[int], y_values: List[Numeric], var: str = "x") -> str:
    """
    Returns a human-readable symbolic string for the Lagrange interpolating polynomial.
    Expands the polynomial into a sum of coefficient * x^k terms (collected form).
    """
    n = len(x_nodes)

    # Represent the polynomial as a list of coefficients [a0, a1, ..., a_{n-1}]
    # where p(x) = a0 + a1*x + a2*x^2 + ...
    # We accumulate coefficients via polynomial multiplication.

    def poly_mul(p: List[float], q: List[float]) -> List[float]:
        """Multiply two polynomials represented as coefficient lists (index = degree)."""
        result = [0.0] * (len(p) + len(q) - 1)
        for i, pi in enumerate(p):
            for j, qj in enumerate(q):
                result[i + j] += pi * qj
        return result

    coeffs = [0.0] * n  # accumulated result polynomial

    for i in range(n):
        xi, yi = x_nodes[i], y_values[i]
        # Build L_i(x) = product_{j != i} (x - xj) / (xi - xj)
        li_poly = [1.0]  # start with constant 1
        denom = 1.0
        for j in range(n):
            if i != j:
                xj = x_nodes[j]
                # multiply by (x - xj): polynomial [-xj, 1]
                li_poly = poly_mul(li_poly, [-xj, 1.0])
                denom *= (xi - xj)
        # scale by yi / denom
        scale = yi / denom
        for k in range(len(li_poly)):
            if k < n:
                coeffs[k] += li_poly[k] * scale

    # Round and format coefficients: integers as int, rationals as fraction, else float
    from fractions import Fraction

    def _fmt(c: float):
        c = round(c, 9)
        if abs(c - round(c)) < 1e-9:
            return int(round(c))
        # Try to express as a simple fraction (denominator <= 1000)
        try:
            frac = Fraction(c).limit_denominator(1000)
            if abs(float(frac) - c) < 1e-9:
                return frac  # keeps exact display via str(frac) e.g. "1/3"
        except Exception:
            pass
        return round(c, 6)

    coeffs = [_fmt(c) for c in coeffs]

    # Build human-readable string
    def _coeff_str(c, with_var: bool) -> str:
        """Format coefficient for display, wrapping fractions in parens when next to a variable."""
        from fractions import Fraction
        if isinstance(c, Fraction):
            return f"({c})" if with_var else str(c)
        return str(c)

    terms = []
    use_math_style = (var == "x")

    for k in range(n - 1, -1, -1):  # highest degree first
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

    # Join terms, handling sign of subsequent terms
    result = terms[0]
    for t in terms[1:]:
        if t.startswith("-"):
            result += f" - {t[1:]}"
        else:
            result += f" + {t}"
    return result


def _approx_eq(a: Any, b: Any, rel_tol: float = 1e-7, abs_tol: float = 1e-9) -> bool:
    """Checks approximate equality for numeric types (int, float, complex)."""
    if a is None or b is None:
        return False
    if isinstance(a, complex) or isinstance(b, complex):
        ca, cb = complex(a), complex(b)
        return (
            math.isclose(ca.real, cb.real, rel_tol=rel_tol, abs_tol=abs_tol)
            and math.isclose(ca.imag, cb.imag, rel_tol=rel_tol, abs_tol=abs_tol)
        )
    try:
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
    except Exception:
        return a == b


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
    Solves linear Diophantine membership for two range arithmetic sequences in O(log(min(step_a, step_b))):
    a + b = target or a - b = target, where
    a = s_a + u * step_a (0 <= u < n_a)
    b = s_b + v * step_b (0 <= v < n_b)
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
        if all(isinstance(x, (int, float)) for x in elems):
            step = (elems[-1] - elems[0]) / (n - 1)
            if all(abs(elems[i] - (elems[0] + i * step)) < 1e-9 for i in range(n)):
                return (elems[0], step, n)
    if unc.d == (1,):
        v = unc.evaluate_at_index((1,))
        if isinstance(v, (int, float)):
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
            real_elems = [
                e.real if isinstance(e, complex) else e
                for e in unc.elements
                if isinstance(e, (int, float, complex))
            ]
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
                if min_r <= 0 <= max_r:
                    bounds = (None, None)
                else:
                    divs = [min_l / min_r, min_l / max_r, max_l / min_r, max_l / max_r]
                    bounds = (min(divs), max(divs))
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
    Core AST Equation Solver engine:
    Recursively inverts operators on the formula AST tree to test whether 'target'
    belongs to the uncertain number in O(depth) / sub-polynomial time.
    """
    if depth > 1000:
        return False
    if memo is None:
        memo = set()

    # Memoization key to avoid re-evaluating duplicate sub-equations
    if isinstance(target, bool):
        sig_key = target
    elif isinstance(target, int):
        sig_key = target
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

    # 1. Interval bounding branch pruning (O(1))
    b = _get_node_bounds(unc)
    if b[0] is not None and b[1] is not None and isinstance(target, (int, float)):
        if (target < b[0] - abs_tol - rel_tol * abs(b[0])) or (target > b[1] + abs_tol + rel_tol * abs(b[1])):
            return False

    ast_type = unc.ast.get("type")

    # 2. Leaf Node Equation Solving
    if ast_type == "leaf":
        # Range representation: O(1) arithmetic inversion
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

        # Explicit elements (list, tuple, set)
        if hasattr(unc, "elements") and unc.elements is not None:
            elems = unc.elements
            if isinstance(elems, set):
                if target in elems:
                    return True
                return any(_approx_eq(e, target, rel_tol, abs_tol) for e in elems)
            elif isinstance(elems, (list, tuple)):
                if len(elems) == 0:
                    return False
                if all(isinstance(x, (int, float)) for x in elems) and isinstance(target, (int, float)):
                    # Binary search in sorted element list
                    pos = bisect.bisect_left(elems, target - abs_tol)
                    if pos < len(elems) and _approx_eq(elems[pos], target, rel_tol, abs_tol):
                        return True
                    if pos > 0 and _approx_eq(elems[pos - 1], target, rel_tol, abs_tol):
                        return True
                    return False
                return any(_approx_eq(e, target, rel_tol, abs_tol) for e in elems)

        # Single scalar scenario d = (1,)
        if unc.d == (1,):
            v = unc.evaluate_at_index((1,))
            return _approx_eq(v, target, rel_tol, abs_tol)

        # Fallback single variable generative function
        if len(unc.d) == 1:
            n = unc.d[0]
            if n <= 100:
                return any(_approx_eq(unc.f(i), target, rel_tol, abs_tol) for i in range(1, n + 1))

        return False

    # 3. Pointwise Operation Node ((o)_1 / (o)_1')
    elif unc.ast.get("space_type") == "pointwise":
        n = unc.d[0] if unc.d else 1
        # Try symbolic solver via SymPy first for pointwise formulas
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

        # Fast scan if n is moderate
        if n <= 10000:
            return any(_approx_eq(unc.evaluate_at_index((i,)), target, rel_tol, abs_tol) for i in range(1, n + 1))
        return False

    # 4. Minkowski Operation Node ((o)_m / AST Operator)
    elif ast_type == "op":
        left: "UncertainNumber" = unc.ast["left"]
        right: "UncertainNumber" = unc.ast["right"]
        op_sym = unc.ast.get("operator_symbol", "+")

        # Fast-path 4a: Both children are linear progressions -> Linear Diophantine Equation in O(log n)
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

        # Fast-path 4b: Invert operation across children
        len_l = left.index_length if left.d else 0
        len_r = right.index_length if right.d else 0

        # Helper to extract small candidate sets
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
                        req_l = target / vr
                        if _solve_ast_membership(left, req_l, rel_tol, abs_tol, memo, depth + 1):
                            return True
                elif op_sym == "/":
                    if not _approx_eq(vr, 0, rel_tol, abs_tol):
                        req_l = target * vr
                        if _solve_ast_membership(left, req_l, rel_tol, abs_tol, memo, depth + 1):
                            return True
                elif op_sym == "**":
                    if _approx_eq(vr, 0, rel_tol, abs_tol):
                        if _approx_eq(target, 1, rel_tol, abs_tol) and left.index_length > 0:
                            return True
                    else:
                        try:
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
                        req_r = target / vl
                        if _solve_ast_membership(right, req_r, rel_tol, abs_tol, memo, depth + 1):
                            return True
                elif op_sym == "/":
                    if _approx_eq(target, 0, rel_tol, abs_tol):
                        if _approx_eq(vl, 0, rel_tol, abs_tol):
                            if any(not _approx_eq(vr, 0, rel_tol, abs_tol) for vr in cands_r or [1]):
                                return True
                    else:
                        req_r = vl / target
                        if not _approx_eq(req_r, 0, rel_tol, abs_tol):
                            if _solve_ast_membership(right, req_r, rel_tol, abs_tol, memo, depth + 1):
                                return True
                elif op_sym == "**":
                    if vl > 0 and not _approx_eq(vl, 1, rel_tol, abs_tol) and target > 0:
                        try:
                            req_r = math.log(target) / math.log(vl)
                            if _solve_ast_membership(right, req_r, rel_tol, abs_tol, memo, depth + 1):
                                return True
                        except Exception:
                            pass

        return False

    return False


class UncertainNumber:
    """
    Represents an Uncertain Number U(K) defined strictly via its Canonical Form (f_X, d_X).
    The index domain 'd' is a Tuple of integers where each element represents 
    the maximum index size (n_i) for that dimension.
    Joined via Abstract AST Composition '*'.
    """

    def __init__(
        self,
        data: Union[List[Numeric], Set[Numeric], None] = None,
        generative_fn: Union[Callable, None] = None,
        index_domain: Union[Tuple[int, ...], None] = None,
        ast_node: Union[dict, None] = None,
    ):
        """
        Initializer for an Uncertain Number:
        1. From raw numeric data: Constructs Lagrange polynomial generative function f(x).
        2. From pure Canonical Form: Generative function 'f' and index domain tuple 'd' representing sizes.
        """
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
                    # Đơn thức bậc nhất O(1): f(x) = start + (x - 1) * step
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
                    y_values = sorted(data_list)
                except Exception:
                    y_values = data_list
                n = len(y_values)

                # 'd' is a tuple containing the maximum index size: d = (n,)
                self.d: Tuple[int, ...] = (n,)
                self.elements = y_values

                is_numeric = all(isinstance(v, (int, float, complex)) and not isinstance(v, bool) for v in y_values)
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
                        # Kiểm tra cấp số cộng để dùng đơn thức bậc nhất f(x) = a*x + b (O(1))
                        step = (y_values[-1] - y_values[0]) / (n - 1)
                        is_arithmetic = False
                        if n <= 100:
                            is_arithmetic = all(abs(y_values[i] - (y_values[0] + i * step)) < 1e-9 for i in range(n))
                        else:
                            is_arithmetic = (
                                abs(y_values[1] - (y_values[0] + step)) < 1e-9 and
                                abs(y_values[n // 2] - (y_values[0] + (n // 2) * step)) < 1e-9 and
                                abs(y_values[-1] - (y_values[0] + (n - 1) * step)) < 1e-9
                            )

                        if is_arithmetic:
                            y0 = y_values[0]
                            # Đơn thức bậc nhất: f(x) = y0 + (x - 1) * step
                            self.f = lambda x: y0 + ((x[0] if isinstance(x, tuple) else x) - 1) * step
                            if step == 1 and y0 == 1:
                                self._formula_template = lambda v: f"{v}"
                            elif step == 1 and y0 == 0:
                                self._formula_template = lambda v: f"{v} - 1"
                            elif step == 1:
                                self._formula_template = lambda v: f"{y0 - 1} + {v}" if y0 > 1 else f"{v} - {1 - y0}"
                            else:
                                self._formula_template = lambda v: f"{y0} + ({v} - 1) * {step}"
                        elif n <= 10:
                            # Lagrange cho tập nhỏ
                            x_nodes = list(range(1, n + 1))
                            self.f = lagrange_interpolation(x_nodes, y_values)
                            # Capture x_nodes/y_values in closure to build symbolic formula
                            _xn, _yv = x_nodes[:], y_values[:]
                            self._formula_template = lambda v, xn=_xn, yv=_yv: lagrange_formula_str(xn, yv, v)
                        else:
                            # Tra cứu mảng O(1) và nội suy tuyến tính từng đoạn
                            def _fast_eval(x: Numeric) -> Numeric:
                                idx_val = x[0] if isinstance(x, tuple) else x
                                if isinstance(idx_val, (int, float)) and 1 <= idx_val <= n:
                                    if isinstance(idx_val, int) or idx_val.is_integer():
                                        return y_values[int(idx_val) - 1]
                                    x_int = int(idx_val)
                                    if x_int == n:
                                        return y_values[-1]
                                    frac = idx_val - x_int
                                    return y_values[x_int - 1] * (1 - frac) + y_values[x_int] * frac
                                return None
                            self.f = _fast_eval
                            self._formula_template = lambda v: f"PiecewiseLinear({v})"
                else:
                    self.f = lambda idx: self.elements[int(idx[0] if isinstance(idx, tuple) else idx) - 1] if self.elements and 1 <= int(idx[0] if isinstance(idx, tuple) else idx) <= len(self.elements) else None
                    self._formula_template = lambda v: f"DiscreteLookup({v})"
                self.ast: dict = {"type": "leaf"}

        elif generative_fn is not None or index_domain is not None or ast_node is not None:
            # Pure Canonical Form initialization with index domain size tuple (f, d)
            self.d = tuple(index_domain) if index_domain is not None else ()
            self.ast = ast_node if ast_node is not None else {"type": "custom_fn"}
            self.f = generative_fn if generative_fn is not None else (lambda idx: self.evaluate_at_index(idx))
            self._formula_template = ast_node.get("formula_template") if ast_node else None

        else:
            raise ValueError(
                "Must provide either raw numeric data OR a generative function 'f' with index domain tuple 'd'."
            )

    def evaluate_at_index(self, index_key: Any) -> Any:
        """
        Evaluates the generative function f_X at a specific index key or index tuple.
        Recursively traverses AST node connections.
        """
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

            # Dynamically execute abstract operator function
            return operator_fn(val_left, val_right)

        return self.f(index_key[0] if len(index_key) == 1 else index_key)

    def to_set(self) -> Set[Any]:
        """
        Executes Lazy Evaluation: Generates candidate index ranges (1..n_i) from tuple 'd'
        and evaluates connected AST nodes to output the final outcome set.
        """
        if self.d == (0,) or (self.ast.get("type") == "leaf" and hasattr(self, "elements") and not self.elements):
            return set()

        # Generate 1-based index ranges for each dimension represented in tuple 'd'
        index_ranges = [range(1, n + 1) for n in self.d]
        
        results = set()
        # Perform Cartesian product iteration over all index dimension ranges
        for idx in itertools.product(*index_ranges):
            try:
                val = self.evaluate_at_index(idx)
                if val is None:
                    continue
                if isinstance(val, float):
                    val = round(val, 10)
                    if val.is_integer():
                        val = int(val)
                elif isinstance(val, complex):
                    real = round(val.real, 10)
                    imag = round(val.imag, 10)
                    if real.is_integer():
                        real = int(real)
                    if imag.is_integer():
                        imag = int(imag)
                    val = complex(real, imag)
                results.add(val)
            except ZeroDivisionError:
                continue
        return results

    # ==================== ABSTRACT AST COMPOSITION OPERATOR '*' ====================

    def compose(self, other: Any, operator_fn: Union[Callable, None] = None) -> "UncertainNumber":
        """
        Abstract composition method: Joins two AST trees under a custom operator function.
        Concatenates index domain tuples d_A and d_B to form combined dimension sizes.
        """
        if not isinstance(other, UncertainNumber):
            other = UncertainNumber([other])

        if operator_fn is None:
            operator_fn = lambda a, b: a * b

        # Concatenate dimension size tuples: d_new = d_A + d_B
        new_d = self.d + other.d

        ast_node = {
            "type": "op",
            "left": self,
            "right": other,
            "operator_fn": operator_fn,
            "operator_symbol": "*",
            "space_type": "minkowski",
        }

        # Return new UncertainNumber holding combined AST root and dimension tuple
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
        """
        Converts a 1D linear integer index into a multi-dimensional scenario index tuple
        corresponding to index domain 'd' = (n_1, n_2, ..., n_k).
        Time complexity: O(k) (Pure Lazy Evaluation - avoids computing full Cartesian sets).
        """
        if not self.d or self.d == (0,):
            return ()

        total_size = math.prod(self.d)
        k0 = flat_idx if zero_based else (flat_idx - 1)

        # Support negative indexing (-1 = last element)
        if k0 < 0:
            k0 += total_size

        if not (0 <= k0 < total_size):
            raise IndexError(f"Index {flat_idx} out of range for UncertainNumber with {total_size} scenarios (domain {self.d}).")

        tuple_idx = []
        # Row-major / odometer decomposition from right to left
        for dim_size in reversed(self.d):
            if dim_size <= 0:
                tuple_idx.append(1)
            else:
                tuple_idx.append((k0 % dim_size) + 1)
                k0 //= dim_size

        return tuple(reversed(tuple_idx))

    def tuple_to_flat_index(self, tuple_idx: Tuple[int, ...], zero_based: bool = True) -> int:
        """
        Converts a multi-dimensional index tuple (1-based per dimension) into a 1D linear integer index.
        Time complexity: O(k).
        """
        if len(tuple_idx) != len(self.d):
            raise ValueError(f"Dimension mismatch: expected {len(self.d)} elements, got {len(tuple_idx)}")

        k0 = 0
        for idx_val, dim_size in zip(tuple_idx, self.d):
            if not (1 <= idx_val <= dim_size):
                raise IndexError(f"Index component {idx_val} out of bounds for dimension size {dim_size}")
            k0 = k0 * dim_size + (idx_val - 1)

        return k0 if zero_based else (k0 + 1)

    def __len__(self) -> int:
        """Returns the total number of scenarios N = prod(d)."""
        return math.prod(self.d) if self.d and self.d != (0,) else 0

    def __iter__(self):
        """Allows direct iteration over the evaluated set elements in sorted order."""
        return iter(sorted(list(self.to_set())))

    def __getitem__(self, index: Any):
        """
        Lazy O(k) scenario indexing:
        - If int: converts flat index -> multi-dimensional tuple index and evaluates f_X(tuple).
        - If tuple: evaluates directly at the given index tuple f_X(tuple).
        """
        if isinstance(index, tuple):
            return self.evaluate_at_index(index)
        elif isinstance(index, int):
            tuple_idx = self.flat_index_to_tuple(index, zero_based=True)
            val = self.evaluate_at_index(tuple_idx)
            if isinstance(val, float):
                val = round(val, 10)
                if val.is_integer():
                    val = int(val)
            return val
        elif isinstance(index, slice):
            total_size = len(self)
            start, stop, step = index.indices(total_size)
            return [self[i] for i in range(start, stop, step)]
        else:
            raise TypeError(f"UncertainNumber indices must be integers or tuples, not {type(index).__name__}")

    def get_formula(self, var_names: Union[List[str], Tuple[str, ...], None] = None) -> str:
        """
        Returns the symbolic mathematical formula f(x_1, x_2, ...) of the generative function.
        """
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
        """Returns the canonical form formula representation: f(x_1, ...) = ..."""
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
                    x.to_set_key() if isinstance(x, UncertainNumber) else x,
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
        """Calculates the total number of scenarios (product of dimension sizes)."""
        return math.prod(self.d)

    def contains(self, target: Any, rel_tol: float = 1e-7, abs_tol: float = 1e-9) -> bool:
        """
        Kiểm tra một phần tử 'target' có thuộc số bất định hay không bằng cách giải phương trình
        nghịch đảo trên cây AST của công thức (AST Equation Solving).
        
        Độ phức tạp: O(k) hoặc O(depth) - Không cần sinh tích Descartes (Lazy Evaluation).
        """
        return _solve_ast_membership(self, target, rel_tol=rel_tol, abs_tol=abs_tol)

    def __contains__(self, item: Any) -> bool:
        """
        Cho phép sử dụng toán tử `in` của Python (ví dụ: `x in U`) để kiểm tra phần tử
        thuộc số bất định thông qua bộ giải phương trình trên cây AST.
        """
        return self.contains(item)

    def solve_equation(
        self,
        target: Any = 0,
        var_names: Union[List[str], Tuple[str, ...], None] = None,
    ) -> List[Tuple[int, ...]]:
        """
        Giải phương trình f(x_1, x_2, ..., x_k) = target trên cây AST của formula.
        Tìm tập các bộ chỉ số kịch bản hợp lệ (i_1, ..., i_k) trong miền d thỏa mãn f(i_1, ...) == target.
        Sử dụng SymPy solver hoặc duyệt trên cây AST kết hợp ràng buộc miền chỉ số.
        """
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

    # ==================== FUNCTIONAL SPACE OPERATORS (LAMBDAS) ====================

    @staticmethod
    def pw(fn: Callable, *args: Any) -> "UncertainNumber":
        """
        Executes a lambda function across arguments in Point-wise Space (o)_1.
        Preserves internal variable identity: all occurrences share the identical point value.
        Example: UncertainNumber.pw(lambda x: x**2 - 3*x + 2, X)
        """
        return pw(fn, *args)

    @staticmethod
    def epw(fn: Callable, *args: Any) -> "UncertainNumber":
        """
        Executes a lambda function across arguments in Extended Point-wise Space (o)_1'.
        Allows broadcasting with scalar values or (1,) domain dimensions.
        Example: UncertainNumber.epw(lambda x, c: x * c, X, 10)
        """
        return epw(fn, *args)

    @staticmethod
    def m(fn: Callable, *args: Any) -> "UncertainNumber":
        """
        Executes a lambda function across arguments in Minkowski Space (o)_m.
        Constructs Cartesian product domain d_1 x d_2 x ... for independent interaction.
        Example: UncertainNumber.m(lambda a, b: a + b, A, B)
        """
        return m(fn, *args)

    @staticmethod
    def em(fn: Callable, *args: Any) -> "UncertainNumber":
        """
        Executes a lambda function across arguments in Extended Minkowski Space (o)_m'.
        Supports inverse equation solving and fractional powers.
        Example: UncertainNumber.em(lambda c, x: c * x, 10, X)
        """
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
    Point-wise Space (o)_1 functional operator for lambdas with UncertainNumber parameters.
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
        elif isinstance(res_expr, (int, float, complex)):
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
    Extended Point-wise Space (o)_1' functional operator for lambdas.
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
        elif isinstance(res_expr, (int, float, complex)):
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
    Minkowski Space (o)_m functional operator for lambdas with UncertainNumber parameters.
    Evaluates algebraic operations across independent Minkowski scenarios.
    """
    if not args:
        return UncertainNumber({fn()})

    unc_args = [_to_unc(a) for a in args]

    # Try evaluating with UncertainNumber instances directly to leverage overloaded Minkowski operators.
    # IMPORTANT: skip this fast-path if fn merely returns one of its input objects (e.g. max/min
    # comparing UncertainNumber objects via __lt__ and returning one of them unchanged).  In that
    # case we must fall through to the Cartesian scalar evaluation so that fn is applied element-wise
    # to produce {f(a, b) : a ∈ A, b ∈ B}.
    try:
        res = fn(*unc_args)
        if isinstance(res, UncertainNumber):
            # If the result is literally one of the input objects, this fast-path is not appropriate.
            if any(res is u for u in unc_args):
                pass  # fall through to Cartesian evaluation below
            else:
                return res
        elif isinstance(res, (int, float, complex, set, list, tuple)):
            return _to_unc(res)
    except Exception:
        pass

    # Fallback to Cartesian scalar evaluation across all input scenarios
    new_d = sum((u.d for u in unc_args), ())

    def generative_fn(idx_tuple: Any) -> Numeric:
        if not isinstance(idx_tuple, tuple):
            idx_tuple = (idx_tuple,)
        arg_vals = []
        curr = 0
        for u in unc_args:
            dim = len(u.d)
            sub_idx = idx_tuple[curr : curr + dim]
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
    Extended Minkowski Space (o)_m' functional operator for lambdas.
    Dispatches to Extended Minkowski equation solving engine for scalars,
    or falls back to Cartesian Minkowski evaluation.

    Supports single-argument lambdas that encode scalar multiplication or
    power operations, e.g.:
        em(lambda A: 0.5 * A, A)   ->  (1/2 * A)_{m'}
        em(lambda A: A ** 0.5, A)  ->  (A^{1/2})_{m'}
    """
    if not args:
        return UncertainNumber({fn()})

    unc_args = [_to_unc(a) for a in args]

    # Check for 2-argument scalar operations suitable for EMinkowskiArithmetic
    if len(unc_args) == 2:
        from .EMinkowskiArithmetic import EMinkowskiArithmetic
        if unc_args[0].d == (1,) or unc_args[1].d == (1,):
            return EMinkowskiArithmetic.em(unc_args[0], unc_args[1], fn)

    # Handle single-argument lambdas of the form fn(A) = scalar * A or fn(A) = A ** scalar.
    # Probe fn with a scalar value to detect the hidden scalar operand.
    if len(unc_args) == 1:
        from .EMinkowskiArithmetic import EMinkowskiArithmetic
        unc_target = unc_args[0]
        probe = 4.0  # arbitrary non-trivial probe value

        # Detect scalar multiplication: fn(x) = c * x  =>  fn(probe)/probe == constant
        try:
            result_mul = fn(probe)
            if isinstance(result_mul, (int, float)) and probe != 0:
                scalar_val = result_mul / probe
                # Verify: fn(probe2) / probe2 == same constant
                probe2 = 9.0
                result_mul2 = fn(probe2)
                if isinstance(result_mul2, (int, float)) and abs(result_mul2 / probe2 - scalar_val) < 1e-9:
                    # Confirmed: fn is a scalar multiplication by scalar_val
                    scalar_unc = UncertainNumber({scalar_val})
                    return EMinkowskiArithmetic.em(scalar_unc, unc_target, lambda x, y: x * y)
        except Exception:
            pass

        # Detect power operation: fn(x) = x ** p  =>  log(fn(probe)) / log(probe) == p
        import math
        try:
            result_pow = fn(probe)
            if isinstance(result_pow, (int, float)) and probe > 0 and result_pow > 0:
                p_val = math.log(result_pow) / math.log(probe)
                probe2 = 9.0
                result_pow2 = fn(probe2)
                if isinstance(result_pow2, (int, float)) and result_pow2 > 0:
                    p_val2 = math.log(result_pow2) / math.log(probe2)
                    if abs(p_val2 - p_val) < 1e-9:
                        # Confirmed: fn is a power operation x ** p_val
                        power_unc = UncertainNumber({p_val})
                        return EMinkowskiArithmetic.em(unc_target, power_unc, lambda x, y: x ** y)
        except Exception:
            pass

    return m(fn, *unc_args)
