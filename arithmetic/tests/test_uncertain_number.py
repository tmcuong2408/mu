import math
import pytest
from arithmetic import UncertainNumber
from arithmetic.UncertainNumber import (
    lagrange_interpolation,
    lagrange_formula_str,
    _extgcd,
    _solve_diophantine_ranges,
    _approx_eq,
)


class TestUncertainNumberInit:
    def test_init_from_set(self):
        u = UncertainNumber({1, 3, 5})
        assert u.d == (3,)
        assert u.to_set() == {1, 3, 5}
        assert len(u) == 3

    def test_init_from_list(self):
        u = UncertainNumber([10, 20, 30, 40])
        assert u.d == (4,)
        assert u.to_set() == {10, 20, 30, 40}

    def test_init_from_range(self):
        r = range(0, 10, 2)
        u = UncertainNumber(r)
        assert u.d == (5,)
        assert u.to_set() == {0, 2, 4, 6, 8}

    def test_init_singleton(self):
        u = UncertainNumber({42})
        assert u.d == (1,)
        assert u.to_set() == {42}
        assert u.evaluate_at_index((1,)) == 42

    def test_init_from_generative_fn(self):
        u = UncertainNumber(generative_fn=lambda idx: (idx[0] if isinstance(idx, tuple) else idx) * 2, index_domain=(4,))
        assert u.d == (4,)
        assert u.to_set() == {2, 4, 6, 8}
        assert u[0] == 2
        assert u[3] == 8


class TestUncertainNumberIndexingAndOdometer:
    def test_odometer_1d(self):
        u = UncertainNumber({10, 20, 30})
        assert u.flat_index_to_tuple(0) == (1,)
        assert u.flat_index_to_tuple(1) == (2,)
        assert u.flat_index_to_tuple(2) == (3,)
        assert u.tuple_to_flat_index((1,)) == 0
        assert u.tuple_to_flat_index((3,)) == 2

    def test_odometer_multi_d(self):
        u = UncertainNumber(generative_fn=lambda idx: idx[0] + idx[1], index_domain=(2, 3))
        assert len(u) == 6
        # d=(2,3) -> (1,1)->0, (1,2)->1, (1,3)->2, (2,1)->3, (2,2)->4, (2,3)->5
        assert u.flat_index_to_tuple(0) == (1, 1)
        assert u.flat_index_to_tuple(2) == (1, 3)
        assert u.flat_index_to_tuple(3) == (2, 1)
        assert u.flat_index_to_tuple(5) == (2, 3)

        assert u.tuple_to_flat_index((1, 1)) == 0
        assert u.tuple_to_flat_index((2, 3)) == 5

    def test_getitem_and_iter(self):
        u = UncertainNumber([100, 200, 300])
        assert u[0] == 100
        assert u[1] == 200
        assert u[-1] == 300
        assert list(iter(u)) == [100, 200, 300]
        assert u[0:2] == [100, 200]

    def test_getitem_index_error(self):
        u = UncertainNumber({1, 2})
        with pytest.raises(IndexError):
            _ = u[5]
        with pytest.raises(IndexError):
            _ = u[-3]


class TestUncertainNumberFormulas:
    def test_formula_linear(self):
        u = UncertainNumber(range(0, 10, 2))
        assert "x" in u.formula

    def test_formula_lagrange(self):
        u = UncertainNumber({1, 4, 9})
        f = u.formula
        assert isinstance(f, str)
        assert len(f) > 0

    def test_str_and_repr(self):
        u = UncertainNumber({1, 2, 3})
        s = str(u)
        r = repr(u)
        assert "{" in s
        assert "_u" in r
        assert "{1, 2, 3}_u" == r

        # Singleton {a}_u biểu diễn thành a (bỏ qua {}_u)
        u_single = UncertainNumber({42})
        assert str(u_single) == "42"
        assert repr(u_single) == "42"

        u_zero = UncertainNumber({0})
        assert str(u_zero) == "0"
        assert repr(u_zero) == "0"


class TestUncertainNumberComparisonAndHash:
    def test_equality(self):
        u1 = UncertainNumber({1, 2, 3})
        u2 = UncertainNumber([1, 2, 3])
        u3 = UncertainNumber({1, 2, 4})
        assert u1 == u2
        assert u1 != u3

    def test_hash_in_set(self):
        u1 = UncertainNumber({1, 2})
        u2 = UncertainNumber([1, 2])
        s = {u1, u2}
        assert len(s) == 1

    def test_lt_ordering(self):
        u1 = UncertainNumber({1, 2})
        u2 = UncertainNumber({2, 3})
        assert u1 < u2 or u2 < u1 or u1 == u2


class TestUncertainNumberMembership:
    def test_exact_membership(self):
        u = UncertainNumber({2, 4, 6, 8})
        assert 2 in u
        assert 4 in u
        assert 5 not in u
        assert 9 not in u

    def test_range_membership(self):
        u = UncertainNumber(range(0, 100, 5))
        assert 0 in u
        assert 25 in u
        assert 95 in u
        assert 96 not in u
        assert -5 not in u
        assert 100 not in u

    def test_float_membership(self):
        u = UncertainNumber([1.0, 2.5, 3.75])
        assert 2.5 in u
        assert 2.50000000001 in u
        assert 4.0 not in u

    def test_extreme_scale_membership(self):
        n_elems = 10**18
        u = UncertainNumber(range(0, 2 * n_elems, 2))
        assert 0 in u
        assert 1_234_567_890_123_456_788 in u
        assert 1_234_567_890_123_456_789 not in u
        assert (2 * n_elems - 2) in u
        assert (2 * n_elems) not in u


class TestUncertainNumberMathUtils:
    def test_lagrange_interpolation_exact(self):
        xs = [1, 2, 3]
        ys = [2, 5, 10]  # y = x^2 + 1
        fn = lagrange_interpolation(xs, ys)
        assert fn(1) == 2
        assert fn(2) == 5
        assert fn(3) == 10
        assert math.isclose(fn(4), 17)

    def test_extgcd(self):
        x, y, g = _extgcd(35, 15)
        assert 35 * x + 15 * y == g
        assert g == 5

    def test_solve_diophantine_ranges(self):
        # A = {0, 2, 4, 6}, B = {1, 3, 5}
        # A + B should contain odd numbers in [1, 11]
        assert _solve_diophantine_ranges(0, 2, 4, 1, 2, 3, "+", 1)  # 0 + 1 = 1
        assert _solve_diophantine_ranges(0, 2, 4, 1, 2, 3, "+", 7)  # 2 + 5 = 7
        assert _solve_diophantine_ranges(0, 2, 4, 1, 2, 3, "+", 11) # 6 + 5 = 11
        assert not _solve_diophantine_ranges(0, 2, 4, 1, 2, 3, "+", 2)
        assert not _solve_diophantine_ranges(0, 2, 4, 1, 2, 3, "+", 13)


class TestSolveEquation:
    def test_solve_equation_pointwise(self):
        u = UncertainNumber({1, 2, 3, 4})
        # f(x) = x, target = 3 -> index 3
        sols = u.solve_equation(3)
        assert (3,) in sols
