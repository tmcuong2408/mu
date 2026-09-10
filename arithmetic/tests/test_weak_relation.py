import pytest
from fractions import Fraction
from arithmetic import UncertainNumber, WeakRelation, weak_relation, mu, s


class TestWeakBinaryRelationTextbookExamples:
    """Kiểm tra các ví dụ và định lý trực tiếp từ tài liệu 'Logic Mở Rộng Và Toán Học Bất Định'."""

    def test_example_2_3(self):
        """
        Ví dụ 2.3:
        Cho A = {1, 2}_u là một số bất định, ta có:
        (A <= A) = 0.75
        (A > A) = 0.25
        Ta nhận xét: mu(A <= A) + mu(A > A) = 1. Mặc dù A = A nhưng về độ lớn thì khác nhau.
        """
        A = UncertainNumber({1, 2})

        le_res = A <= A
        gt_res = A > A

        assert isinstance(le_res, WeakRelation)
        assert isinstance(gt_res, WeakRelation)

        # Kiểm tra giá trị chân lý chính xác
        assert le_res == 0.75
        assert le_res.truth_value == Fraction(3, 4)
        assert gt_res == 0.25
        assert gt_res.truth_value == Fraction(1, 4)

        # mu(A <= A) + mu(A > A) = 1
        assert le_res + gt_res == 1.0

        # Kiểm tra biểu diễn chuỗi dạng con số số học [0, 1]
        assert str(le_res) == "0.75"
        assert repr(le_res) == "0.75"
        assert str(gt_res) == "0.25"
        assert repr(gt_res) == "0.25"

        # Biểu thức chi tiết qua detail()
        assert le_res.detail() == "({1, 2}_u <= {1, 2}_u)_0.75"
        assert gt_res.detail() == "({1, 2}_u > {1, 2}_u)_0.25"

    def test_example_2_4(self):
        """
        Ví dụ 2.4:
        Cho A = {1, 2}_u và B = {3}_u = 3 thì A <= B.
        """
        A = UncertainNumber({1, 2})
        B = UncertainNumber({3})

        # So sánh giữa hai UncertainNumber
        res1 = A <= B
        assert res1 == 1.0
        assert res1.is_certain

        # So sánh trực tiếp với hằng số đơn trị 3 (số thực)
        res2 = A <= 3
        assert res2 == 1.0
        assert res2.is_certain

        # Phản xạ (reflected): 3 >= A
        res3 = 3 >= A
        assert res3 == 1.0

        # A > 3 phải có chân lý bằng 0
        assert (A > 3) == 0.0

    def test_theorem_2_2_equilibrium(self):
        """
        Định lý 2.2 (Cân bằng):
        Với X in U(R) có n phần tử phân biệt thì:
        mu(X <= X) = 1/2 + 1/(2n)
        """
        for n in [2, 3, 4, 5, 10, 50, 100]:
            X = UncertainNumber(set(range(1, n + 1)))
            expected = Fraction(1, 2) + Fraction(1, 2 * n)
            res = X <= X
            assert res.truth_value == expected
            assert res == float(expected)


class TestWeakBinaryRelationAllOperators:
    """Kiểm tra toàn bộ 6 toán tử so sánh hai ngôi yếu."""

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
        # A = {1, 3}, B = {2, 4}
        # Cặp A x B: (1,2), (1,4), (3,2), (3,4) (tổng 4 cặp)
        # <= : (1,2), (1,4), (3,4) -> 3/4 = 0.75
        # <  : (1,2), (1,4), (3,4) -> 3/4 = 0.75
        # >= : (3,2) -> 1/4 = 0.25
        # >  : (3,2) -> 1/4 = 0.25
        # == : không có -> 0.0
        # != : cả 4 cặp -> 1.0
        A = UncertainNumber({1, 3})
        B = UncertainNumber({2, 4})

        assert (A <= B) == 0.75
        assert (A < B) == 0.75
        assert (A >= B) == 0.25
        assert (A > B) == 0.25
        assert (A == B) == 0.0
        assert (A != B) == 1.0

    def test_equality_operator(self):
        # A = {1, 2}, A == A -> (1,1), (2,2) thỏa -> 2/4 = 0.5
        A = UncertainNumber({1, 2})
        assert (A == A) == 0.5
        assert (A != A) == 0.5


class TestClassicalIdentity:
    """Định lý 2.10 (Đồng nhất với số học cổ điển): singleton số bất định tương đương số thực."""

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
    """Định nghĩa 2.17: Quan hệ hai ngôi yếu có trọng số."""

    def test_weighted_relation(self):
        # A = {1, 2}, w_A(1) = 0.2, w_A(2) = 0.8
        # B = {1, 2}, w_B(1) = 0.5, w_B(2) = 0.5
        # Cặp (1,1): 1<=1 -> 0.2 * 0.5 = 0.1
        # Cặp (1,2): 1<=2 -> 0.2 * 0.5 = 0.1
        # Cặp (2,1): 2<=1 -> False
        # Cặp (2,2): 2<=2 -> 0.8 * 0.5 = 0.4
        # mu(A <= B) = 0.1 + 0.1 + 0.4 = 0.6
        A = UncertainNumber({1, 2}, weights={1: Fraction(2, 10), 2: Fraction(8, 10)})
        B = UncertainNumber({1, 2}, weights={1: Fraction(5, 10), 2: Fraction(5, 10)})

        res = A <= B
        assert res == pytest.approx(0.6)
        assert res.truth_value == Fraction(6, 10)


class TestCustomAndConvenienceMethods:
    """Kiểm tra weak_relation, mu và các dạng gọi khác nhau."""

    def test_custom_relation_divisibility(self):
        # Quan hệ b chia hết cho a: b % a == 0
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

        # Cả 2 thứ tự tham số đều được hỗ trợ
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
