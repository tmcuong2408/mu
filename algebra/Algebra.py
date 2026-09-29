from typing import Callable, Any, Optional, Union, List, Tuple
from .UncertainGraph import UncertainGraph, CompleteGraph, pgr, pgr_f, graph, gr
from .Draw import draw, draw_ascii, draw_ast


class Algebra:
    """
    Algebra - Mathematics of Uncertainty
    Central engine for algebraic structures, uncertain functions, graphs, and mappings.
    Based on Chapter 3: Uncertain Algebra in 'Extended Logic and Mathematics of Uncertainty'.
    """

    @staticmethod
    def pgr(func: Callable, input_val: Any, space: str = "m") -> Union[UncertainGraph, CompleteGraph]:
        """
        Computes the Point-wise Graph of an Uncertain Function:
            pgr_f(A) := A × f(A)  (Definition 3.3)
        
        Evaluated lazily via AST; computation occurs only when printed or drawn.
        """
        return pgr(func, input_val, space=space)

    @staticmethod
    def pgr_f(func: Callable, input_val: Any, space: str = "m") -> Union[UncertainGraph, CompleteGraph]:
        """Alias for Algebra.pgr(func, input_val)."""
        return pgr_f(func, input_val, space=space)

    @staticmethod
    def graph(func: Callable, input_val: Any, space: str = "m") -> Union[UncertainGraph, CompleteGraph]:
        """
        General function to compute the graph of an uncertain function.
        Returns a lazy AST-driven UncertainGraph instance.
        """
        return graph(func, input_val, space=space)

    @staticmethod
    def gr(func: Callable, domain: Any, space: str = "m") -> CompleteGraph:
        """
        Computes the Complete Graph of an Uncertain Function over a scenario domain:
            gr(f) := { pgr_f(X) : X in domain }  (Definition 3.4)
        """
        return gr(func, domain, space=space)

    @staticmethod
    def draw(graph_obj: Any, **kwargs: Any) -> Any:
        """Draws the computed uncertain graph using matplotlib with terminal ASCII fallback."""
        return draw(graph_obj, **kwargs)

    @staticmethod
    def draw_ascii(graph_obj: Any, **kwargs: Any) -> str:
        """Renders ASCII visualization of the graph in the terminal."""
        return draw_ascii(graph_obj, **kwargs)

    @staticmethod
    def draw_ast(graph_obj: Any, **kwargs: Any) -> str:
        """Visualizes the AST tree hierarchy of the graph."""
        return draw_ast(graph_obj, **kwargs)
