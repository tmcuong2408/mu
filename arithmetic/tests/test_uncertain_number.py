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

        # Singleton {a}_u represented as a (omits {}_u)
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


class TestUncertainNumberApproximation:
    def test_approximate_floats(self):
        u = UncertainNumber({1.23456, 2.71828})
        res = u.approximate(2)
        assert 1.23 in res
        assert 2.72 in res
        assert len(res) == 2

    def test_approx_and_round_aliases(self):
        u = UncertainNumber({1.23456, 2.71828})
        res1 = u.approximate(2)
        res2 = u.approx(2)
        res3 = u.round(2)
        res4 = round(u, 2)
        assert res1.to_set() == res2.to_set() == res3.to_set() == res4.to_set()

    def test_approximate_integer_ndigits_zero(self):
        u = UncertainNumber({1.2, 2.7})
        res = u.approximate(0)
        assert res.to_set() == {1, 3}
        assert round(u).to_set() == {1, 3}

    def test_approximate_negative_ndigits(self):
        u = UncertainNumber({123, 456})
        res = u.approximate(-1)
        assert res.to_set() == {120, 460}

    def test_approximate_fractions(self):
        from fractions import Fraction
        u = UncertainNumber({Fraction(1, 3), Fraction(2, 3)})
        res = u.approximate(2)
        assert Fraction(33, 100) in res
        assert Fraction(67, 100) in res

    def test_approximate_complex(self):
        u = UncertainNumber({1.234 + 5.678j})
        res = u.approximate(2)
        assert (1.23 + 5.68j) in res

    def test_approximate_with_weights_merging(self):
        u = UncertainNumber({1.234, 1.231, 3.456}, weights={1.234: 0.2, 1.231: 0.3, 3.456: 0.5})
        res = u.approximate(2)
        # 1.234 and 1.231 merge into 1.23 with total weight 0.2 + 0.3 = 0.5
        assert res.mu("<=", 1.25) == pytest.approx(0.5)
        assert res.mu("<=", 4.0) == pytest.approx(1.0)

    def test_approximate_ast_expression(self):
        from arithmetic import pw
        X = UncertainNumber({1, 2, 4})
        Y = pw(lambda x: x / 3, X)
        res = Y.approximate(2)
        assert 0.33 in res
        assert 0.67 in res
        assert 1.33 in res

    def test_module_level_helpers(self):
        from arithmetic import approximate, approx, Arithmetic
        u = UncertainNumber({1.2345, 2.3456})
        r1 = approximate(u, 2)
        r2 = approx(u, 2)
        r3 = Arithmetic.approximate(u, 2)
        r4 = Arithmetic.round(u, 2)
        assert r1.to_set() == r2.to_set() == r3.to_set() == r4.to_set()

    def test_approximate_empty(self):
        u = UncertainNumber(set())
        assert u.approximate(2).to_set() == set()


class TestUncertainNumberCanonicalInit:
    def test_init_with_keyword_f_and_d_multivariable(self):
        # f is a multivariable function taking 2 variables (x1, x2)
        u = UncertainNumber(f=lambda x1, x2: x1 + 2 * x2, d=(2, 3))
        assert u.d == (2, 3)
        assert len(u) == 6
        # Scenarios: (1,1)->3, (1,2)->5, (1,3)->7, (2,1)->4, (2,2)->6, (2,3)->8
        assert u.to_set() == {3, 4, 5, 6, 7, 8}
        assert u[0] == 3  # (1, 1)
        assert u[(1, 2)] == 5
        assert u[(2, 3)] == 8

    def test_init_with_positional_f_and_d(self):
        # f and d passed positionally: UncertainNumber(f, d)
        u = UncertainNumber(lambda x, y: x * y, (3, 2))
        assert u.d == (3, 2)
        assert len(u) == 6
        assert u.to_set() == {1, 2, 3, 4, 6}

    def test_from_canonical_and_from_fd(self):
        u1 = UncertainNumber.from_canonical(lambda x, y: x - y, (2, 2))
        u2 = UncertainNumber.from_fd(lambda x, y: x - y, (2, 2))
        assert u1.d == (2, 2)
        assert u2.d == (2, 2)
        assert u1.to_set() == u2.to_set() == {0, -1, 1}

    def test_direct_function_evaluation_styles(self):
        u = UncertainNumber(f=lambda x1, x2: x1 + 10 * x2, d=(2, 3))
        # Direct call with unpacked arguments
        assert u.f(1, 2) == 21
        # Direct call with single tuple argument
        assert u.f((1, 2)) == 21
        # evaluate_at_index
        assert u.evaluate_at_index((1, 2)) == 21
        assert u.evaluate_at_index([1, 2]) == 21

    def test_tuple_parameter_function(self):
        # Multivariable function defined with a single tuple parameter idx
        u = UncertainNumber(f=lambda idx: idx[0] ** 2 + idx[1], d=(3, 2))
        assert u.f(2, 1) == 5
        assert u.f((2, 1)) == 5
        assert u[(2, 1)] == 5
        assert u.to_set() == {2, 3, 5, 6, 10, 11}

    def test_varargs_function(self):
        # Function taking *args
        u = UncertainNumber(f=lambda *args: sum(args), d=(2, 2, 2))
        assert u.d == (2, 2, 2)
        assert len(u) == 8
        assert u.f(1, 2, 2) == 5
        assert u.f((1, 2, 2)) == 5
        assert min(u.to_set()) == 3  # (1, 1, 1)
        assert max(u.to_set()) == 6  # (2, 2, 2)

    def test_1d_canonical_function(self):
        # 1D with scalar argument
        u1 = UncertainNumber(f=lambda x: 3 * x + 1, d=(4,))
        assert u1.to_set() == {4, 7, 10, 13}
        assert u1.f(2) == 7
        assert u1.f((2,)) == 7

        # 1D with integer d (auto-converted to tuple (3,))
        u2 = UncertainNumber(f=lambda idx: idx[0] * 5, d=3)
        assert u2.d == (3,)
        assert u2.to_set() == {5, 10, 15}

    def test_canonical_with_arithmetic_spaces(self):
        from arithmetic import Arithmetic
        A = UncertainNumber(f=lambda x, y: x + y, d=(2, 2))  # {2, 3, 4}
        B = UncertainNumber(f=lambda z: 10 * z, d=(2,))       # {10, 20}

        # Minkowski space addition
        C_m = Arithmetic.add(A, B, space="m")
        assert C_m.d == (2, 2, 2)
        assert C_m.to_set() == {12, 13, 14, 22, 23, 24}

        # Pointwise space addition
        C_pw = Arithmetic.add(A, A, space="pw")
        assert C_pw.d == (2, 2)
        assert C_pw.to_set() == {4, 6, 8}

    def test_canonical_weak_relation(self):
        A = UncertainNumber(f=lambda x: x, d=(2,))
        B = UncertainNumber(f=lambda x: x, d=(2,))
        assert A.mu("<=", B) == 0.75

    def test_invalid_domain_raises(self):
        with pytest.raises(ValueError):
            UncertainNumber(f=lambda x: x, d=(-1,))
        with pytest.raises(ValueError):
            UncertainNumber(f=lambda x: x, d=(2, "invalid"))
        with pytest.raises(TypeError):
            UncertainNumber(f=lambda x: x, d=object())

