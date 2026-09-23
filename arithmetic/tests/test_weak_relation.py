import pytest
from fractions import Fraction
from arithmetic import UncertainNumber, WeakRelation, weak_relation, mu, s


class TestWeakBinaryRelationTextbookExamples:
    """Tests examples and theorems directly from 'Extended Logic and Mathematics of Uncertainty'."""

    def test_example_2_3(self):
        """
        Example 2.3:
        Let A = {1, 2}_u be an uncertain number, we have:
        (A <= A) = 0.75
        (A > A) = 0.25
        Notice: mu(A <= A) + mu(A > A) = 1. Although A = A, their relative magnitudes differ.
        """
        A = UncertainNumber({1, 2})

        le_res = A <= A
        gt_res = A > A

        assert isinstance(le_res, WeakRelation)
        assert isinstance(gt_res, WeakRelation)

        # Verify exact truth values
        assert le_res == 0.75
        assert le_res.truth_value == Fraction(3, 4)
        assert gt_res == 0.25
        assert gt_res.truth_value == Fraction(1, 4)

        # mu(A <= A) + mu(A > A) = 1
        assert le_res + gt_res == 1.0

        # Verify numeric string representation in [0, 1]
        assert str(le_res) == "0.75"
        assert repr(le_res) == "0.75"
        assert str(gt_res) == "0.25"
        assert repr(gt_res) == "0.25"

        # Detailed expression via detail()
        assert le_res.detail() == "({1, 2}_u <= {1, 2}_u)_0.75"
        assert gt_res.detail() == "({1, 2}_u > {1, 2}_u)_0.25"

    def test_example_2_4(self):
        """
        Example 2.4:
        For A = {1, 2}_u and B = {3}_u = 3, then A <= B.
        """
        A = UncertainNumber({1, 2})
        B = UncertainNumber({3})

        # Comparison between two UncertainNumbers
        res1 = A <= B
        assert res1 == 1.0
        assert res1.is_certain

        # Direct comparison with scalar constant 3 (real number)
        res2 = A <= 3
        assert res2 == 1.0
        assert res2.is_certain

        # Reflected comparison: 3 >= A
        res3 = 3 >= A
        assert res3 == 1.0

        # A > 3 must evaluate to 0 truth value
        assert (A > 3) == 0.0

    def test_theorem_2_2_equilibrium(self):
        """
        Theorem 2.2 (Equilibrium):
        For X in U(R) having n distinct elements:
        mu(X <= X) = 1/2 + 1/(2n)
        """
        for n in [2, 3, 4, 5, 10, 50, 100]:
            X = UncertainNumber(set(range(1, n + 1)))
            expected = Fraction(1, 2) + Fraction(1, 2 * n)
            res = X <= X
            assert res.truth_value == expected
            assert res == float(expected)


class TestWeakBinaryRelationAllOperators:
    """Tests all 6 weak binary comparison operators."""

    def test_operators_duality(self):
        A = UncertainNumber({1, 3, 5})
        B = UncertainNumber({2, 4})

        # mu(A <= B) + mu(A > B) == 1
        assert (A <= B) + (A > B) == 1.0

        # mu(A < B) + mu(A >= B) == 1
        assert (A < B) + (A >= B) == 1.0

        # mu(A == B) + mu(A != B) == 1
        assert (A == B) + (A != B) == 1.0

    def test_pairwise_counts_verification(self):
        # Pairs A x B: (1,2), (1,4), (3,2), (3,4) (total 4 pairs)
        # <= : (1,2), (1,4), (3,4) -> 3/4 = 0.75
        # <  : (1,2), (1,4), (3,4) -> 3/4 = 0.75
        # >= : (3,2) -> 1/4 = 0.25
        # >  : (3,2) -> 1/4 = 0.25
        # == : none -> 0.0
        # != : all 4 pairs -> 1.0
        A = UncertainNumber({1, 3})
        B = UncertainNumber({2, 4})

        assert (A <= B) == 0.75
        assert (A < B) == 0.75
        assert (A >= B) == 0.25
        assert (A > B) == 0.25
        assert (A == B) == 0.0
        assert (A != B) == 1.0

    def test_equality_operator(self):
        # A = {1, 2}, A == A -> (1,1), (2,2) satisfied -> 2/4 = 0.5
        A = UncertainNumber({1, 2})
        assert (A == A) == 0.5
        assert (A != A) == 0.5


class TestClassicalIdentity:
    """Theorem 2.10 (Classical Identity): singleton uncertain numbers are equivalent to real numbers."""

    def test_singletons(self):
        a = UncertainNumber({5})
        b = UncertainNumber({10})

        assert (a <= b) == 1.0
        assert (a < b) == 1.0
        assert (a >= b) == 0.0
        assert (a > b) == 0.0
        assert (a == b) == 0.0
        assert (a != b) == 1.0

        # a == a
        assert (a == a) == 1.0


class TestWeightedWeakRelation:
    """Definition 2.17: Weighted weak binary relation."""

    def test_weighted_relation(self):
        # A = {1, 2}, w_A(1) = 0.2, w_A(2) = 0.8
        # B = {1, 2}, w_B(1) = 0.5, w_B(2) = 0.5
        # Pair (1,1): 1<=1 -> 0.2 * 0.5 = 0.1
        # Pair (1,2): 1<=2 -> 0.2 * 0.5 = 0.1
        # Pair (2,1): 2<=1 -> False
        # Pair (2,2): 2<=2 -> 0.8 * 0.5 = 0.4
        # mu(A <= B) = 0.1 + 0.1 + 0.4 = 0.6
        A = UncertainNumber({1, 2}, weights={1: Fraction(2, 10), 2: Fraction(8, 10)})
        B = UncertainNumber({1, 2}, weights={1: Fraction(5, 10), 2: Fraction(5, 10)})

        res = A <= B
        assert res == pytest.approx(0.6)
        assert res.truth_value == Fraction(6, 10)


class TestCustomAndConvenienceMethods:
    """Tests weak_relation, mu and various calling conventions."""

    def test_custom_relation_divisibility(self):
        # Relation b divisible by a: b % a == 0
        A = UncertainNumber({2, 3})
        B = UncertainNumber({6, 7})
        # (2, 6): True
        # (2, 7): False
        # (3, 6): True
        # (3, 7): False
        # -> 2/4 = 0.5
        divisible_fn = lambda a, b: b % a == 0
        res = A.weak_relation(B, divisible_fn)
        assert res == 0.5

    def test_mu_alias_and_arg_order(self):
        A = UncertainNumber({1, 2})
        B = UncertainNumber({2, 3})

        # Both parameter orders are supported
        assert A.mu(B, "<=") == A.weak_relation(B, "<=")
        assert A.mu("<=", B) == A.weak_relation(B, "<=")

        # Module-level functions
        assert weak_relation(A, B, "<=") == (A <= B)
        assert mu(A, B, "<=") == (A <= B)

    def test_is_identical(self):
        u1 = UncertainNumber({1, 2, 3})
        u2 = UncertainNumber([1, 2, 3])
        u3 = UncertainNumber({1, 2, 4})

        assert u1.is_identical(u2)
        assert not u1.is_identical(u3)
