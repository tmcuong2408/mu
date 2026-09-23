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
    s,
    c,
)


class Test_Complex_Functions:
    def test_1(self):
        f_1 = m(lambda x:x+x)
        f_2 = pw(lambda x:x+x)
        a = s(1,2)
        b = f_2(f_1(a))
        assert b.to_set() == {4, 6, 8}

    def test_2(self):
        a = s(1,2)
        f_1 = m(lambda x:x+x)
        f_2 = pw(lambda x:x+x)
        b = f_2(f_1(a))
        assert b.to_set() == {4, 6, 8}
    
    def test_3(self):
        a = s(1,2)
        f_1 = m(lambda x:x+x)
        f_2 = pw(lambda x:x+x)
        b = f_1(f_2(a))
        assert b.to_set() == {4, 6, 8}
    
    def test_4(self):
        a = s(1,2,4)
        f_1 = m(lambda x:x+x)
        f_2 = pw(lambda x:x+x)
        b = f_1(f_2(a))
        assert b.to_set() == {4, 6, 10, 8, 12, 16}


class Test_Composition_c:
    def test_c_two_functions(self):
        # c([f_2, f_1]) = f_2 ∘ f_1, i.e. f_2(f_1(a))
        a = s(1, 2)
        f_1 = m(lambda x: x + x)
        f_2 = pw(lambda x: x + x)
        h = c([f_2, f_1])
        assert h(a).to_set() == {4, 6, 8}

    def test_c_two_functions_reversed(self):
        # c([f_1, f_2]) = f_1 ∘ f_2, i.e. f_1(f_2(a))
        a = s(1, 2)
        f_1 = m(lambda x: x + x)
        f_2 = pw(lambda x: x + x)
        h = c([f_1, f_2])
        assert h(a).to_set() == {4, 6, 8}

    def test_c_three_functions(self):
        # c([f_3, f_2, f_1]) = f_3(f_2(f_1(a)))
        a = s(1, 2)
        f_1 = m(lambda x: x + x)
        f_2 = pw(lambda x: x + x)
        f_3 = m(lambda x: x + 1)
        h = c([f_3, f_2, f_1])
        assert h(a).to_set() == f_3(f_2(f_1(a))).to_set()

    def test_c_single_function(self):
        # c([f]) with single element is identity composition
        a = s(1, 2, 3)
        f = pw(lambda x: x * 2)
        h = c([f])
        assert h(a).to_set() == f(a).to_set()

    def test_c_reusable(self):
        # returned callable can be reused on multiple inputs
        f_1 = m(lambda x: x + x)
        f_2 = pw(lambda x: x + x)
        h = c([f_2, f_1])
        a = s(1, 2)
        b = s(1, 2, 4)
        assert h(a).to_set() == {4, 6, 8}
        assert h(b).to_set() == {4, 6, 10, 8, 12, 16}

    def test_c_empty_list_raises(self):
        with pytest.raises(ValueError):
            c([])

    def test_c_non_callable_raises(self):
        with pytest.raises(TypeError):
            c([42])