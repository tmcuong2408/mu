import itertools
import math
from fractions import Fraction
from typing import Callable, Union, List, Tuple, Set, Any, Optional

from arithmetic import UncertainNumber, pw, m, epw, em
from arithmetic.UncertainNumber import _safe_round_val


def _ensure_uncertain(val: Any) -> UncertainNumber:
    """Converts raw collection, numeric value, or range to UncertainNumber."""
    if isinstance(val, UncertainNumber):
        return val
    if isinstance(val, (set, list, tuple, range)):
        return UncertainNumber(val)
    return UncertainNumber({val})


class UncertainGraph(UncertainNumber):
    """
    Representation of the Graph of an Uncertain Function at a point/set:
        pgr_f(A) := A × f(A)  (Definition 3.3 in Extended Logic & Mathematics of Uncertainty)
    
    AST Structure & Lazy Evaluation:
        - The graph maintains an AST tree connecting the input domain A and function evaluation f(A).
        - Computations are strictly lazy: scenario point combinations are NOT evaluated upon creation.
        - Calculations only take place when requested (e.g., when calling print(), str(), repr(),
          to_set(), or draw()).
    """

    def __init__(
        self,
        func: Callable,
        input_uncertain: Any,
        space: str = "m",
    ):
        self.func = func
        self.input = _ensure_uncertain(input_uncertain)
        self.space = space.lower() if isinstance(space, str) else "m"
        self._is_computed: bool = False
        self._cached_points: Optional[Set[Tuple[Any, Any]]] = None

        # Build output UncertainNumber f(A) using appropriate arithmetic space
        # Note: f(A) returns an AST-based UncertainNumber which is also lazily evaluated!
        if self.space in ("pw", "pointwise"):
            self.space_type = "pointwise"
            try:
                self.output = pw(func, self.input)
            except Exception:
                self.output = func(self.input)
        elif self.space in ("epw", "extended_pointwise"):
            self.space_type = "pointwise"
            try:
                self.output = epw(func, self.input)
            except Exception:
                self.output = func(self.input)
        elif self.space in ("em", "extended_minkowski"):
            self.space_type = "minkowski"
            try:
                self.output = em(func, self.input)
            except Exception:
                self.output = func(self.input)
        else:  # "m" / "minkowski" (Default)
            self.space_type = "minkowski"
            try:
                # Attempt direct application first (e.g. lambda X: X + X)
                res = func(self.input)
                if isinstance(res, UncertainNumber):
                    self.output = res
                else:
                    self.output = UncertainNumber({res})
            except Exception:
                # Fallback to Minkowski lifting m(func, input)
                self.output = m(func, self.input)

        # Ensure output is an UncertainNumber
        if not isinstance(self.output, UncertainNumber):
            self.output = _ensure_uncertain(self.output)

        # Build the Graph AST Node: pgr_f(A) = A × f(A)
        ast_node = {
            "type": "pgr",
            "func": func,
            "left": self.input,
            "right": self.output,
            "input_ast": getattr(self.input, "ast", {"type": "leaf"}),
            "output_ast": getattr(self.output, "ast", {"type": "leaf"}),
            "space_type": self.space_type,
            "operator_symbol": "×",
        }

        # Determine Scenario Index Domain
        if self.space_type == "pointwise":
            # Pointwise space (o)_1 preserves point identity: d_pgr = d_A
            new_d = self.input.d
            def generative_fn(idx_tuple: Any) -> Tuple[Any, Any]:
                if not isinstance(idx_tuple, tuple):
                    idx_tuple = (idx_tuple,)
                x = self.input.evaluate_at_index(idx_tuple)
                y = self.output.evaluate_at_index(idx_tuple)
                return (x, y)
        else:
            # Minkowski space (o)_m Cartesian interaction: d_pgr = d_A × d_f(A)
            new_d = self.input.d + self.output.d
            in_dim = len(self.input.d)
            def generative_fn(idx_tuple: Any) -> Tuple[Any, Any]:
                if not isinstance(idx_tuple, tuple):
                    idx_tuple = (idx_tuple,)
                idx_x = idx_tuple[:in_dim]
                idx_y = idx_tuple[in_dim:]
                x = self.input.evaluate_at_index(idx_x)
                y = self.output.evaluate_at_index(idx_y)
                return (x, y)

        # Initialize parent UncertainNumber with lazy generative function and AST
        super().__init__(
            generative_fn=generative_fn,
            index_domain=new_d,
            ast_node=ast_node,
        )

    @property
    def is_computed(self) -> bool:
        """Returns True if the graph has been evaluated, False if still lazy."""
        return self._is_computed

    def to_set(self) -> Set[Tuple[Any, Any]]:
        """
        Triggers evaluation of graph scenario coordinates from AST and index domain.
        Marks the graph as evaluated and caches the result.
        """
        if self._cached_points is not None:
            self._is_computed = True
            return self._cached_points

        self._is_computed = True

        if self.d == (0,) or (self.ast.get("type") == "leaf" and hasattr(self, "elements") and not self.elements):
            self._cached_points = set()
            return self._cached_points

        index_ranges = [range(1, n + 1) for n in self.d]
        results = set()

        for idx in itertools.product(*index_ranges):
            try:
                pair = self.evaluate_at_index(idx)
                if pair is not None and isinstance(pair, tuple) and len(pair) == 2:
                    x, y = pair
                    x_clean = _safe_round_val(x)
                    y_clean = _safe_round_val(y)
                    results.add((x_clean, y_clean))
            except ZeroDivisionError:
                continue

        self._cached_points = results
        return results

    @property
    def points(self) -> List[Tuple[Any, Any]]:
        """Returns a sorted list of scenario coordinate pairs (x, y). Triggers computation."""
        pts = self.to_set()
        return sorted(
            list(pts),
            key=lambda p: (
                float(p[0].real) if isinstance(p[0], complex) else float(p[0]) if isinstance(p[0], (int, float, Fraction)) else 0,
                float(p[1].real) if isinstance(p[1], complex) else float(p[1]) if isinstance(p[1], (int, float, Fraction)) else 0,
                str(p),
            ),
        )

    @property
    def domain_values(self) -> List[Any]:
        """Returns the unique sorted x-coordinates present in the graph."""
        pts = self.to_set()
        xs = set(p[0] for p in pts)
        return sorted(
            list(xs),
            key=lambda x: (
                float(x.real) if isinstance(x, complex) else float(x) if isinstance(x, (int, float, Fraction)) else 0,
                str(x),
            ),
        )

    @property
    def image_values(self) -> List[Any]:
        """Returns the unique sorted y-coordinates present in the graph."""
        pts = self.to_set()
        ys = set(p[1] for p in pts)
        return sorted(
            list(ys),
            key=lambda y: (
                float(y.real) if isinstance(y, complex) else float(y) if isinstance(y, (int, float, Fraction)) else 0,
                str(y),
            ),
        )

    def __repr__(self) -> str:
        """
        Triggers evaluation and returns canonical uncertain set notation:
            {(x1, y1), (x2, y2), ...}_u
        """
        pts = self.points
        if not pts:
            return "{}_u"
        inner = ", ".join(f"({x}, {y})" for x, y in pts)
        return f"{{{inner}}}_u"

    def __str__(self) -> str:
        return self.__repr__()

    def draw(self, **kwargs) -> Any:
        """Draws this graph using the draw module."""
        from .Draw import draw
        return draw(self, **kwargs)

    def draw_ascii(self, **kwargs) -> str:
        """Renders an ASCII visualization of this graph in the terminal."""
        from .Draw import draw_ascii
        return draw_ascii(self, **kwargs)

    def draw_ast(self, **kwargs) -> Any:
        """Visualizes the AST tree structure of this uncertain graph."""
        from .Draw import draw_ast
        return draw_ast(self, **kwargs)


class CompleteGraph:
    """
    Representation of the Complete Graph of an Uncertain Function over a domain:
        gr(f) := { pgr_f(X) : X in X_domain }  (Definition 3.4 in Extended Logic & Mathematics of Uncertainty)
    
    AST Structure & Lazy Evaluation:
        - Maintains AST references to each sub-graph pgr_f(X) across the scenario domain.
        - Only evaluates when printed or explicitly drawn.
    """

    def __init__(
        self,
        func: Callable,
        domain: Union[List[Any], Set[Any], Tuple[Any, ...]],
        space: str = "m",
    ):
        self.func = func
        self.domain = list(domain)
        self.space = space
        self._is_computed: bool = False
        self._cached_graphs: Optional[List[UncertainGraph]] = None

        # Build AST node for complete graph
        self.ast = {
            "type": "gr",
            "func": func,
            "space_type": space,
            "domain_count": len(self.domain),
        }

    @property
    def is_computed(self) -> bool:
        return self._is_computed

    @property
    def subgraphs(self) -> List[UncertainGraph]:
        """Returns the list of UncertainGraph objects for each X in domain."""
        if self._cached_graphs is None:
            self._cached_graphs = [
                UncertainGraph(self.func, X, space=self.space)
                for X in self.domain
            ]
        return self._cached_graphs

    def to_list(self) -> List[Set[Tuple[Any, Any]]]:
        """Evaluates all subgraphs and returns a list of scenario coordinate sets."""
        self._is_computed = True
        return [g.to_set() for g in self.subgraphs]

    def all_points(self) -> List[Tuple[Any, Any]]:
        """Returns all coordinate pairs across all subgraphs."""
        self._is_computed = True
        pts = set()
        for g in self.subgraphs:
            pts.update(g.to_set())
        return sorted(
            list(pts),
            key=lambda p: (
                float(p[0].real) if isinstance(p[0], complex) else float(p[0]) if isinstance(p[0], (int, float, Fraction)) else 0,
                float(p[1].real) if isinstance(p[1], complex) else float(p[1]) if isinstance(p[1], (int, float, Fraction)) else 0,
                str(p),
            ),
        )

    def __repr__(self) -> str:
        self._is_computed = True
        sub_reprs = [repr(g) for g in self.subgraphs]
        inner = ", ".join(sub_reprs)
        return f"{{{inner}}}_u"

    def __str__(self) -> str:
        return self.__repr__()

    def __iter__(self):
        return iter(self.subgraphs)

    def __len__(self) -> int:
        return len(self.domain)

    def draw(self, **kwargs) -> Any:
        from .Draw import draw
        return draw(self, **kwargs)

    def draw_ascii(self, **kwargs) -> str:
        from .Draw import draw_ascii
        return draw_ascii(self, **kwargs)

    def draw_ast(self, **kwargs) -> Any:
        from .Draw import draw_ast
        return draw_ast(self, **kwargs)


# ==================== CONVENIENCE FUNCTIONAL ALIASES ====================

def pgr(func: Callable, input_val: Any, space: str = "m") -> Union[UncertainGraph, CompleteGraph]:
    """
    Computes the Point-wise Graph of an Uncertain Function:
        pgr_f(A) := A × f(A)  (Definition 3.3)
    
    If input_val is a collection of UncertainNumbers (domain X),
    returns the complete graph gr(f).
    
    Calculation is strictly lazy using AST: evaluation occurs only when printed or drawn.
    """
    if isinstance(input_val, (list, tuple)) and input_val and all(isinstance(x, (UncertainNumber, set)) for x in input_val):
        return CompleteGraph(func, input_val, space=space)
    return UncertainGraph(func, input_val, space=space)


def pgr_f(func: Callable, input_val: Any, space: str = "m") -> Union[UncertainGraph, CompleteGraph]:
    """Alias for pgr(f, A)."""
    return pgr(func, input_val, space=space)


def graph(func: Callable, input_val: Any, space: str = "m") -> Union[UncertainGraph, CompleteGraph]:
    """
    General graph computation function:
        pgr_f(A) := A × f(A)
    
    Strictly lazy with AST representation; evaluates only upon print or draw.
    """
    return pgr(func, input_val, space=space)


def gr(func: Callable, domain: Any, space: str = "m") -> CompleteGraph:
    """
    Computes the Complete Graph of an Uncertain Function over a domain:
        gr(f) := { pgr_f(X) : X in domain }  (Definition 3.4)
    """
    if isinstance(domain, UncertainNumber):
        domain_list = [UncertainNumber({val}) for val in domain.to_set()]
    elif isinstance(domain, (list, tuple, set)):
        domain_list = list(domain)
    else:
        domain_list = [domain]
    return CompleteGraph(func, domain_list, space=space)


Graph = UncertainGraph
