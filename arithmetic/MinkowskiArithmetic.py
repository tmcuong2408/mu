import itertools
from typing import Callable, Union, List, Tuple, Set, Any

# Type alias for numeric types (supporting both Real and Complex numbers)
Numeric = Union[int, float, complex]

from .UncertainNumber import UncertainNumber  # <--- Đảm bảo có từ khóa này

class MinkowskiArithmetic:
    """
    Engine for Minkowski Arithmetic Space (o)_m[cite: 1].
    Models independent scenario interactions via Cartesian product domain d_A x d_B[cite: 1].
    """

    @staticmethod
    def m(
        left: UncertainNumber,
        right: UncertainNumber,
        operator_fn: Callable[[Numeric, Numeric], Numeric],
        operator_symbol: str = "+",
    ) -> UncertainNumber:
        """
        Connects two AST nodes in Minkowski Space (o)_m[cite: 1].
        Concatenates index domain tuples: d_new = d_A + d_B[cite: 1].
        """
        new_d = left.d + right.d

        ast_node = {
            "type": "op",
            "left": left,
            "right": right,
            "operator_fn": operator_fn,
            "operator_symbol": operator_symbol,
            "space_type": "minkowski",
        }

        res = UncertainNumber(
            generative_fn=lambda idx: res.evaluate_at_index(idx),
            index_domain=new_d,
            ast_node=ast_node,
        )
        return res