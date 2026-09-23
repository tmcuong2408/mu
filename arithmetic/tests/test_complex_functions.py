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
    s
)

class Test_Complex_Functions:
    def test_1(self):
        f_1 = m(lambda x:x+x)
        f_2 = pw(lambda x:x+x)
        a = s(1,2)
        b = f_2(f_1(a))
        assert b.to_set() == {4, 6, 8}