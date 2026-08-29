import itertools
from fractions import Fraction
from typing import Callable, Union, List, Tuple, Set, Any

from .UncertainNumber import UncertainNumber, _approx_eq, _safe_round_val
from .MinkowskiArithmetic import MinkowskiArithmetic

# Type alias for numeric types
Numeric = Union[int, float, complex, Fraction]


def _exact_add_set(s: Set[Numeric], other: Set[Numeric]) -> Set[Numeric]:
    """Tính Minkowski sum của hai tập, ưu tiên int/Fraction."""
    result = set()
    for x in s:
        for y in other:
            v = x + y
            result.add(_safe_round_val(v))
    return result


def _exact_mul_set(s: Set[Numeric], other: Set[Numeric]) -> Set[Numeric]:
    """Tính Minkowski product của hai tập, ưu tiên int/Fraction."""
    result = set()
    for x in s:
        for y in other:
            v = x * y
            result.add(_safe_round_val(v))
    return result


def _minkowski_sum_n(unc: UncertainNumber, n: int) -> UncertainNumber:
    """Computes (A^n_+)_m = A +_m A +_m ... +_m A (n times)."""
    if n <= 0:
        return UncertainNumber({0})
    s = unc.to_set()
    cur = set(s)
    for _ in range(n - 1):
        cur = _exact_add_set(cur, s)
    return UncertainNumber(cur)


def _minkowski_mul_n(unc: UncertainNumber, n: int) -> UncertainNumber:
    """Computes (A^n_*)_m = A *_m A *_m ... *_m A (n times)."""
    if n <= 0:
        return UncertainNumber({1})
    s = unc.to_set()
    cur = set(s)
    for _ in range(n - 1):
        cur = _exact_mul_set(cur, s)
    return UncertainNumber(cur)


def _solve_inverse_minkowski_sum(b_set: Set[Numeric], q: int) -> Union[Set[Numeric], None]:
    """
    Solves for X such that (X^q_+)_m == b_set (Inverse Minkowski Sum).
    Definition 3.10: (1/q A)_{m'} := { X : (X^q_+)_m = A }_u.
    Sử dụng Fraction để đảm bảo độ chính xác tuyệt đối.
    """
    if q == 1:
        return b_set
    if not b_set:
        return set()

    # Chuyển sang Fraction để so sánh chính xác
    b_set_f = set()
    for b in b_set:
        try:
            b_set_f.add(Fraction(b))
        except (TypeError, ValueError):
            b_set_f.add(b)

    sorted_b = sorted(list(b_set_f))
    b_min = sorted_b[0]
    b_max = sorted_b[-1]

    # x_min = b_min / q, x_max = b_max / q — chính xác với Fraction
    x_min = Fraction(b_min, q)
    x_max = Fraction(b_max, q)

    # Chuyển về int nếu là số nguyên
    if x_min.denominator == 1:
        x_min = x_min.numerator
    if x_max.denominator == 1:
        x_max = x_max.numerator

    candidates = set()
    for b in sorted_b:
        cand = b - (q - 1) * x_min
        cand = _safe_round_val(cand)
        if x_min <= cand <= x_max:
            candidates.add(cand)

    candidates.add(x_min)
    candidates.add(x_max)
    cand_list = sorted(list(candidates))

    def m_sum_set_exact(s: Set[Numeric], count: int) -> Set[Numeric]:
        cur = set(s)
        for _ in range(count - 1):
            cur = _exact_add_set(cur, s)
        return cur

    def _normalize_set(s: Set[Numeric]) -> Set:
        return {_safe_round_val(x) for x in s}

    b_norm = _normalize_set(b_set_f)

    # 1. Quick check: full candidate set
    if _normalize_set(m_sum_set_exact(set(cand_list), q)) == b_norm:
        return {_safe_round_val(c) for c in cand_list}

    # 2. Subset search
    other_cands = [c for c in cand_list if c != x_min and c != x_max]
    for r in range(len(other_cands), -1, -1):
        for combo in itertools.combinations(other_cands, r):
            sub = {x_min, x_max} | set(combo)
            if _normalize_set(m_sum_set_exact(sub, q)) == b_norm:
                return {_safe_round_val(c) for c in sub}

    return None


def _int_nth_root(n: int, p: int) -> Union[int, None]:
    """
    Tính căn bậc p của n một cách chính xác (trả về int nếu là số nguyên chính xác).
    Dùng Newton method với số nguyên để tránh lỗi float.
    """
    if n < 0:
        return None
    if n == 0:
        return 0
    if n == 1:
        return 1
    if p == 1:
        return n
    if p == 2:
        # Dùng integer square root
        import math
        g = math.isqrt(n)
        if g * g == n:
            return g
        return None
    # Newton's method cho n^(1/p)
    import math
    g = int(round(n ** (1.0 / p)))
    for candidate in range(max(0, g - 2), g + 3):
        if candidate ** p == n:
            return candidate
    return None


def _solve_all_inverse_minkowski_mul(b_set: Set[Numeric], q: int) -> List[UncertainNumber]:
    """
    Solves for all X such that (X^q_*)_m == b_set (Definition 3.11).
    Dùng integer arithmetic chính xác khi có thể.
    """
    if q == 1:
        return [UncertainNumber(b_set)]
    if not b_set:
        return []

    # Kiểm tra xem tất cả có phải số nguyên dương không
    b_list = list(b_set)
    all_positive_int = all(isinstance(b, int) and b > 0 for b in b_list)
    all_positive = all(
        (isinstance(b, (int, Fraction)) and b > 0) or
        (isinstance(b, float) and b > 0)
        for b in b_list
    )

    if not all_positive:
        return []

    sorted_b = sorted(b_list)
    solutions = []

    if all_positive_int:
        # Chính xác hoàn toàn với số nguyên
        b_min = sorted_b[0]
        b_max = sorted_b[-1]

        x_min = _int_nth_root(b_min, q)
        x_max = _int_nth_root(b_max, q)

        if x_min is None or x_max is None:
            # Không phải perfect power — thử brute force với Fraction
            pass
        else:
            cands = set()
            for b in sorted_b:
                # b / x_min^(q-1)
                if x_min > 0:
                    c_frac = Fraction(b, x_min ** (q - 1))
                    c = _safe_round_val(c_frac)
                    if isinstance(c, int) and x_min <= c <= x_max:
                        cands.add(c)
            cands.add(x_min)
            cands.add(x_max)
            cand_list = sorted(list(cands))

            def m_mul_set_exact(s: Set[Numeric], count: int) -> Set[Numeric]:
                cur = set(s)
                for _ in range(count - 1):
                    cur = _exact_mul_set(cur, s)
                return cur

            def _norm(s):
                return frozenset(_safe_round_val(x) for x in s)

            b_norm = _norm(b_set)

            other_cands = [c for c in cand_list if c != x_min and c != x_max]
            for r in range(len(other_cands) + 1):
                for combo in itertools.combinations(other_cands, r):
                    sub = {x_min, x_max} | set(combo)
                    if _norm(m_mul_set_exact(sub, q)) == b_norm:
                        solutions.append(UncertainNumber({_safe_round_val(c) for c in sub}))

            if q % 2 == 0:
                for sol in list(solutions):
                    neg_set = {-x for x in sol.to_set()}
                    if _norm(m_mul_set_exact(neg_set, q)) == b_norm:
                        if _norm(neg_set) not in [_norm(s.to_set()) for s in solutions]:
                            solutions.append(UncertainNumber(neg_set))

            return solutions

    # Fallback với float cho số không nguyên
    sorted_b_f = [float(b) for b in sorted_b]
    b_min_f = sorted_b_f[0]
    b_max_f = sorted_b_f[-1]
    x_min_f = b_min_f ** (1.0 / q)
    x_max_f = b_max_f ** (1.0 / q)

    cands = set()
    for b_f in sorted_b_f:
        if x_min_f > 0:
            c = b_f / (x_min_f ** (q - 1))
            c = _safe_round_val(c)
            if x_min_f <= float(c) <= x_max_f:
                cands.add(c)
    cands.add(_safe_round_val(x_min_f))
    cands.add(_safe_round_val(x_max_f))
    cand_list = sorted(list(cands))

    def m_mul_float(s: Set[Numeric], count: int) -> Set[Numeric]:
        cur = set(s)
        for _ in range(count - 1):
            cur = _exact_mul_set(cur, s)
        return cur

    def _norm_f(s):
        return frozenset(_safe_round_val(x) for x in s)

    b_set_norm = frozenset(_safe_round_val(b) for b in b_set)

    other_cands = [c for c in cand_list if c not in (_safe_round_val(x_min_f), _safe_round_val(x_max_f))]
    for r in range(len(other_cands) + 1):
        for combo in itertools.combinations(other_cands, r):
            sub = {_safe_round_val(x_min_f), _safe_round_val(x_max_f)} | set(combo)
            if _norm_f(m_mul_float(sub, q)) == b_set_norm:
                solutions.append(UncertainNumber(sub))

    if q % 2 == 0:
        for sol in list(solutions):
            neg_set = {-x for x in sol.to_set()}
            if _norm_f(m_mul_float(neg_set, q)) == b_set_norm:
                if _norm_f(neg_set) not in [_norm_f(s.to_set()) for s in solutions]:
                    solutions.append(UncertainNumber(neg_set))

    return solutions


class EMinkowskiArithmetic:
    """
    Engine for Extended Minkowski Arithmetic Space (o)_m'.
    Extends (o)_m with fractional powers, ratio spaces, and inverse equation solvers.
    Definition 3.10: (p/q A)_{m'} := { X : (X^q_+)_m = (A^p_+)_m }_u
    Definition 3.11: (A^{p/q})_{m'} := { X : (X^q_*)_m = (A^p_*)_m }_u
    Hỗ trợ số nguyên siêu lớn qua Fraction và int arithmetic chính xác.
    """

    @staticmethod
    def em(
        left: UncertainNumber,
        right: UncertainNumber,
        operator_fn: Callable[[Numeric, Numeric], Numeric],
    ) -> UncertainNumber:
        """Connects AST nodes or solves inverse relations in Extended Minkowski Space (o)_m'."""
        # 1. Detect power operations in (o)_m' (Definition 3.11)
        is_pow = False
        try:
            if operator_fn(2, 3) == 8 and operator_fn(4, Fraction(1, 2)) == 2:
                is_pow = True
        except Exception:
            pass
        if not is_pow:
            try:
                if operator_fn(2, 3) == 8 and operator_fn(4, 0.5) == 2.0:
                    is_pow = True
            except Exception:
                pass

        if is_pow and right.d == (1,):
            scalar_val = list(right.to_set())[0]
            unc_target = left

            try:
                frac = Fraction(scalar_val).limit_denominator(10**9)
                p, q = frac.numerator, frac.denominator
            except Exception:
                p, q = int(scalar_val), 1

            if q == 1 and p >= 0:
                return _minkowski_mul_n(unc_target, p)
            elif q > 1 and p > 0:
                target_b = _minkowski_mul_n(unc_target, p).to_set()
                solutions = _solve_all_inverse_minkowski_mul(target_b, q)
                if solutions:
                    return UncertainNumber(solutions)
                else:
                    return UncertainNumber(set())

        # 2. Detect scalar multiplication in (o)_m' (Definition 3.10)
        is_mul = False
        try:
            if operator_fn(2, 3) == 6 and operator_fn(0, 5) == 0:
                is_mul = True
        except Exception:
            pass

        if is_mul and (left.d == (1,) or right.d == (1,)):
            if left.d == (1,) and right.d != (1,):
                scalar_val = list(left.to_set())[0]
                unc_target = right
            elif right.d == (1,) and left.d != (1,):
                scalar_val = list(right.to_set())[0]
                unc_target = left
            else:
                return MinkowskiArithmetic.m(left, right, operator_fn)

            try:
                frac = Fraction(scalar_val).limit_denominator(10**9)
                p, q = frac.numerator, frac.denominator
            except Exception:
                p, q = int(scalar_val), 1

            if q == 1 and p >= 0:
                return _minkowski_sum_n(unc_target, p)
            elif q > 1 and p > 0:
                target_b = _minkowski_sum_n(unc_target, p).to_set()
                x_sol = _solve_inverse_minkowski_sum(target_b, q)
                if x_sol is not None:
                    return UncertainNumber(x_sol)
                else:
                    return UncertainNumber(set())

        # Default fallback to Minkowski space AST connection
        return MinkowskiArithmetic.m(left, right, operator_fn)