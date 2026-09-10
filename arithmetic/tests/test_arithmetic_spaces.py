import pytest
from arithmetic import (
    UncertainNumber,
    Arithmetic,
    PointwiseArithmetic,
    EPointwiseArithmetic,
    MinkowskiArithmetic,
    EMinkowskiArithmetic,
    pw,
    epw,
    m,
    em,
)


class TestPointwiseArithmetic:
    def test_pw_basic_operators(self):
        X = UncertainNumber({1, 2, 3, 4})
        # X - X in pointwise space should equal 0 for all scenarios
        res_sub = Arithmetic.sub(X, X, space="pw")
        assert res_sub.d == (4,)
        assert res_sub.to_set() == {0}

        # X + X in pointwise space should equal {2, 4, 6, 8}
        res_add = Arithmetic.add(X, X, space="pw")
        assert res_add.to_set() == {2, 4, 6, 8}

    def test_pw_domain_mismatch_raises(self):
        A = UncertainNumber({1, 2})
        B = UncertainNumber({1, 2, 3})
        with pytest.raises(ValueError):
            Arithmetic.add(A, B, space="pw")

    def test_pw_lambda_functional(self):
        X = UncertainNumber({1, 2, 3, 4})
        # f(x) = x^2 + 5x + 6: for x=1->12, x=2->20, x=3->30, x=4->42
        f_pw = pw(lambda x: x**2 + 5*x + 6, X)
        assert f_pw.to_set() == {12, 20, 30, 42}
        assert f_pw.d == (4,)

    def test_pw_identity_cancellation(self):
        X = UncertainNumber({10, 20, 30})
        zero_pw = pw(lambda x: x - x, X)
        assert zero_pw.to_set() == {0}


class TestExtendedPointwiseArithmetic:
    def test_epw_scalar_broadcast(self):
        X = UncertainNumber({1, 2, 3, 4})
        res_epw = Arithmetic.mul(X, 10, space="epw")
        assert res_epw.to_set() == {10, 20, 30, 40}
        assert res_epw.d == (4,)

        res_epw_rev = Arithmetic.mul(10, X, space="epw")
        assert res_epw_rev.to_set() == {10, 20, 30, 40}

    def test_epw_lambda_functional(self):
        X = UncertainNumber({1, 2, 3, 4})
        res = epw(lambda x, c: x * c + 1, X, 10)
        assert res.to_set() == {11, 21, 31, 41}


class TestMinkowskiArithmetic:
    def test_m_addition_cartesian(self):
        A = UncertainNumber({1, 3})
        B = UncertainNumber({10, 20})
        # A + B = {11, 21, 13, 23}
        C = A + B
        assert C.d == (2, 2)
        assert len(C) == 4
        assert C.to_set() == {11, 13, 21, 23}
        assert 11 in C
        assert 23 in C
        assert 10 not in C

    def test_m_subtraction_inflation(self):
        X = UncertainNumber({1, 2, 3, 4})
        # In Minkowski space, X - X inflates to {-3, -2, -1, 0, 1, 2, 3}
        res_m = Arithmetic.sub(X, X, space="m")
        assert res_m.d == (4, 4)
        assert res_m.to_set() == {-3, -2, -1, 0, 1, 2, 3}

    def test_m_all_operators(self):
        A = UncertainNumber({2, 4})
        B = UncertainNumber({1, 2})

        # Multiplication
        assert (A * B).to_set() == {2, 4, 8}
        # True division
        assert (A / B).to_set() == {1.0, 2.0, 4.0}
        # Floor division
        assert (A // B).to_set() == {1, 2, 4}
        # Modulo
        assert (A % B).to_set() == {0}
        # Power
        assert (A ** B).to_set() == {2, 4, 16}
        # Unary negation
        assert (-A).to_set() == {-4, -2}
        # Unary positive
        assert (+A).to_set() == {2, 4}
        # Absolute value
        neg_u = UncertainNumber({-5, 3})
        assert abs(neg_u).to_set() == {3, 5}

    def test_m_lambda_functional(self):
        A = UncertainNumber({1, 3})
        B = UncertainNumber({10, 20})
        res_m_fn = m(lambda a, b: a + b, A, B)
        assert res_m_fn.to_set() == {11, 13, 21, 23}


class TestExtendedMinkowskiArithmetic:
    def test_em_fractional_scalar_mul_valid(self):
        # 0.5 * {2, 3, 4} solves for X such that X + X = {2, 3, 4}
        # Solution X = {1, 2} since {1, 2} + {1, 2} = {2, 3, 4}
        Y = UncertainNumber({2, 3, 4})
        res_em_half = Arithmetic.mul(0.5, Y, space="em")
        assert res_em_half.to_set() == {1, 2}

    def test_em_fractional_scalar_mul_no_solution(self):
        # Set with no inverse Minkowski sum decomposition -> returns empty set
        X_custom = UncertainNumber({1, 5, 6, 7, 25, 30, 35, 36, 42, 49})
        res_em_none = Arithmetic.mul(0.5, X_custom, space="em")
        assert res_em_none.to_set() == set()

    def test_em_fractional_power_root(self):
        # {1, 2, 4}^0.5 solves for X such that X * X = {1, 2, 4}
        # X = {1, 2} and {-1, -2}
        Y = UncertainNumber({1, 2, 4})
        res_sqrt = Arithmetic.pow(Y, 0.5, space="em")
        # May return a collection of solutions
        assert len(res_sqrt.to_set()) > 0
        all_vals = set()
        for elem in res_sqrt:
            if isinstance(elem, UncertainNumber):
                all_vals.update(elem.to_set())
            else:
                all_vals.add(elem)
        assert 1 in all_vals or -1 in all_vals

    def test_em_integer_multiplier(self):
        # 2 * X in em is X +_m X
        X = UncertainNumber({1, 2})
        res = Arithmetic.mul(2, X, space="em")
        assert res.to_set() == {2, 3, 4}


class TestArithmeticDispatch:
    def test_invalid_space_raises(self):
        A = UncertainNumber({1, 2})
        B = UncertainNumber({3, 4})
        with pytest.raises(ValueError):
            Arithmetic.add(A, B, space="invalid_space")
