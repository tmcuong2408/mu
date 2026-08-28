import itertools
from typing import Callable, Union, List, Tuple, Set, Any

# Type alias for numeric types (supporting both Real and Complex numbers)
Numeric = Union[int, float, complex]
from .UncertainNumber import UncertainNumber  # <--- Đảm bảo có từ khóa này
class EPointwiseArithmetic:
    """
    Engine for Extended Point-wise Arithmetic Space (o)_1'[cite: 1].
    Extends (o)_1 with hierarchical alignment, inverse operators, and nested structural mappings[cite: 1].
    """

    @staticmethod
    def epw(
        left: UncertainNumber,
        right: UncertainNumber,
        operator_fn: Callable[[Numeric, Numeric], Numeric],
        operator_symbol: str = "+",
    ) -> UncertainNumber:
        """Connects AST nodes in Extended Point-wise Space (o)_1'[cite: 1]."""
        # Broadcasts index domains if one operand is scalar dimension (1,)
        if left.d != right.d:
            if left.d == (1,):
                new_d = right.d
            elif right.d == (1,):
                new_d = left.d
            else:
                raise ValueError(f"Incompatible domains for (o)_1': {left.d} and {right.d}")
        else:
            new_d = left.d

        ast_node = {
            "type": "op",
            "left": left,
            "right": right,
            "operator_fn": operator_fn,
            "operator_symbol": operator_symbol,
            "space_type": "pointwise",
        }

        res = UncertainNumber(
            generative_fn=lambda idx: res.evaluate_at_index(idx),
            index_domain=new_d,
            ast_node=ast_node,
        )
        return res