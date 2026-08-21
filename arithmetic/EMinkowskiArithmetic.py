import itertools
from fractions import Fraction
from typing import Callable, Union, List, Tuple, Set, Any

from .UncertainNumber import UncertainNumber
from .MinkowskiArithmetic import MinkowskiArithmetic

# Type alias for numeric types (supporting both Real and Complex numbers)
Numeric = Union[int, float, complex]


def _minkowski_sum_n(unc: UncertainNumber, n: int) -> UncertainNumber:
    """Computes (A^n_+)_m = A +_m A +_m ... +_m A (n times) efficiently."""
    if n <= 0:
        return UncertainNumber({0})
    s = unc.to_set()
    cur = set(s)
    for _ in range(n - 1):
        cur = {round(x + y, 10) if isinstance(x + y, float) else x + y for x in cur for y in s}
        cur = {int(x) if isinstance(x, float) and x.is_integer() else x for x in cur}
    return UncertainNumber(cur)


def _minkowski_mul_n(unc: UncertainNumber, n: int) -> UncertainNumber:
    """Computes (A^n_*)_m = A *_m A *_m ... *_m A (n times) efficiently."""
    if n <= 0:
        return UncertainNumber({1})
    s = unc.to_set()
    cur = set(s)
    for _ in range(n - 1):
        cur = {round(x * y, 10) if isinstance(x * y, float) else x * y for x in cur for y in s}
        cur = {int(x) if isinstance(x, float) and x.is_integer() else x for x in cur}
    return UncertainNumber(cur)


def _solve_inverse_minkowski_sum(b_set: Set[Numeric], q: int) -> Union[Set[Numeric], None]:
    """
    Solves for X such that (X^q_+)_m == b_set (Inverse Minkowski Sum).
    Definition 3.10: (1/q A)_{m'} := { X : (X^q_+)_m = A }_u.
    """
    if q == 1:
        return b_set
    if not b_set:
        return set()

    sorted_b = sorted(list(b_set))
    b_min = sorted_b[0]
    b_max = sorted_b[-1]

    x_min = b_min / q
    x_max = b_max / q
    if isinstance(b_min, int) and b_min % q == 0 and isinstance(b_max, int) and b_max % q == 0:
        x_min = int(x_min)
        x_max = int(x_max)

    candidates = set()
    for b in sorted_b:
        cand = b - (q - 1) * x_min
        if isinstance(cand, float) and cand.is_integer():
            cand = int(cand)
        if x_min <= cand <= x_max:
            candidates.add(cand)

    candidates.add(x_min)
    candidates.add(x_max)
    cand_list = sorted(list(candidates))

    def m_sum_set(s: Set[Numeric], count: int) -> Set[Numeric]:
        cur = s
        for _ in range(count - 1):
            cur = {round(x + y, 10) if isinstance(x + y, float) else x + y for x in cur for y in s}
            cur = {int(x) if isinstance(x, float) and x.is_integer() else x for x in cur}
        return cur

    # 1. Quick check: full candidate set
    if m_sum_set(set(cand_list), q) == b_set:
        return set(cand_list)

    # 2. Subset search
    other_cands = [c for c in cand_list if c != x_min and c != x_max]
    for r in range(len(other_cands), -1, -1):
        for combo in itertools.combinations(other_cands, r):
            sub = {x_min, x_max} | set(combo)
            if m_sum_set(sub, q) == b_set:
                return sub

    return None


def _solve_all_inverse_minkowski_mul(b_set: Set[Numeric], q: int) -> List[UncertainNumber]:
    """
    Solves for all X such that (X^q_*)_m == b_set (Definition 3.11).
    For example, (sqrt({1, 2, 4}))_m = {{1, 2}_u, {-1, -2}_u}_u.
    """
    if q == 1:
        return [UncertainNumber(b_set)]
    if not b_set:
        return []

    sorted_b = sorted(list(b_set))
    solutions = []

    # Check if all elements in b_set are positive
    if all(b > 0 for b in sorted_b):
        b_min = sorted_b[0]
        b_max = sorted_b[-1]
        x_min = b_min ** (1.0 / q)
        x_max = b_max ** (1.0 / q)
        if isinstance(b_min, int) and isinstance(x_min, float) and x_min.is_integer():
            x_min = int(x_min)
        if isinstance(b_max, int) and isinstance(x_max, float) and x_max.is_integer():
            x_max = int(x_max)

        # Generate candidates
        cands = set()
        for b in sorted_b:
            c = b / (x_min ** (q - 1))
            if isinstance(c, float) and c.is_integer():
                c = int(c)
            if x_min <= c <= x_max:
                cands.add(c)
        cands.add(x_min)
        cands.add(x_max)
        cand_list = sorted(list(cands))

        def m_mul_set(s: Set[Numeric], count: int) -> Set[Numeric]:
            cur = s
            for _ in range(count - 1):
                cur = {round(x * y, 10) if isinstance(x * y, float) else x * y for x in cur for y in s}
                cur = {int(x) if isinstance(x, float) and x.is_integer() else x for x in cur}
            return cur

        other_cands = [c for c in cand_list if c != x_min and c != x_max]
        for r in range(len(other_cands) + 1):
            for combo in itertools.combinations(other_cands, r):
                sub = {x_min, x_max} | set(combo)
                if m_mul_set(sub, q) == b_set:
                    solutions.append(UncertainNumber(sub))

        if q % 2 == 0:
            for sol in list(solutions):
                neg_set = {-x for x in sol.to_set()}
                if m_mul_set(neg_set, q) == b_set and neg_set not in [s.to_set() for s in solutions]:
                    solutions.append(UncertainNumber(neg_set))

    return solutions


class EMinkowskiArithmetic:
    """
    Engine for Extended Minkowski Arithmetic Space (o)_m'.
    Extends (o)_m with fractional powers, ratio spaces, and inverse equation solvers.
    Definition 3.10: (p/q A)_{m'} := { X : (X^q_+)_m = (A^p_+)_m }_u
    Definition 3.11: (A^{p/q})_{m'} := { X : (X^q_*)_m = (A^p_*)_m }_u
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
            if operator_fn(2, 3) == 8 and operator_fn(4, 0.5) == 2:
                is_pow = True
        except Exception:
            pass

        if is_pow and right.d == (1,):
            scalar_val = list(right.to_set())[0]
            unc_target = left

            try:
                frac = Fraction(scalar_val).limit_denominator(1000)
                p, q = frac.numerator, frac.denominator
            except Exception:
                p, q = int(scalar_val), 1

            if q == 1 and p >= 0:
                return _minkowski_mul_n(unc_target, p)
            elif q > 1 and p > 0:
                # Target B = (A^p_*)_m
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
                # Both are scalars
                return MinkowskiArithmetic.m(left, right, operator_fn)

            # Parse scalar as fraction p/q
            try:
                frac = Fraction(scalar_val).limit_denominator(1000)
                p, q = frac.numerator, frac.denominator
            except Exception:
                p, q = int(scalar_val), 1

            if q == 1 and p >= 0:
                # (p A)_{m'} = (A^p_+)_m = A +_m ... +_m A (p times)
                return _minkowski_sum_n(unc_target, p)
            elif q > 1 and p > 0:
                # First compute target set B = (A^p_+)_m
                target_b = _minkowski_sum_n(unc_target, p).to_set()
                # Solve (X^q_+)_m = B
                x_sol = _solve_inverse_minkowski_sum(target_b, q)
                if x_sol is not None:
                    return UncertainNumber(x_sol)
                else:
                    return UncertainNumber(set())

        # Default fallback to Minkowski space AST connection
        return MinkowskiArithmetic.m(left, right, operator_fn)