import itertools
from typing import Callable, Union, List, Tuple, Set, Any

# Type alias for numeric types (supporting both Real and Complex numbers)
Numeric = Union[int, float, complex]

from .UncertainNumber import UncertainNumber  # <--- Đảm bảo có từ khóa này
class PointwiseArithmetic:
    """
    Engine for Point-wise Arithmetic Space (o)_1[cite: 1].
    Binds AST nodes under internal variable identity constraints (shared single index)[cite: 1].
    """

    @staticmethod
    def pw(
        left: UncertainNumber,
        right: UncertainNumber,
        operator_fn: Callable[[Numeric, Numeric], Numeric],
    ) -> UncertainNumber:
        """
        Connects two AST nodes in Point-wise Space (o)_1[cite: 1].
        Enforces identical index domain sizes to preserve algebraic identity[cite: 1].
        """
        if left.d != right.d:
            raise ValueError(
                f"Point-wise Space (o)_1 requires identical index domains. Got {left.d} and {right.d}."
            )

        # Domain size remains identical: d_new = d_A = d_B[cite: 1]
        new_d = left.d

        ast_node = {
            "type": "op",
            "left": left,
            "right": right,
            "operator_fn": operator_fn,
            "space_type": "pointwise",
        }

        res = UncertainNumber(
            generative_fn=lambda idx: res.evaluate_at_index(idx),
            index_domain=new_d,
            ast_node=ast_node,
        )
        return res