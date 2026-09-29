"""
Algebra - Mathematics of Uncertainty
Algebraic structures for uncertain numbers, functions, and arithmetic spaces.
Author: Tran Manh Cuong (tmcuong2408@gmail.com)
"""

from .UncertainGraph import (
    UncertainGraph,
    CompleteGraph,
    Graph,
    pgr,
    pgr_f,
    graph,
    gr,
)
from .Draw import draw, draw_ascii, draw_ast
from .Algebra import Algebra

# Attach .graph() and .pgr() methods to UncertainNumber for convenience
try:
    from arithmetic import UncertainNumber

    def _unc_graph(self, func, space="m"):
        return pgr(func, self, space=space)

    def _unc_pgr(self, func, space="m"):
        return pgr(func, self, space=space)

    UncertainNumber.graph = _unc_graph
    UncertainNumber.pgr = _unc_pgr
except ImportError:
    pass

__version__ = "0.1.0"

__all__ = [
    "UncertainGraph",
    "CompleteGraph",
    "Graph",
    "Algebra",
    "pgr",
    "pgr_f",
    "graph",
    "gr",
    "draw",
    "draw_ascii",
    "draw_ast",
    "__version__",
]
