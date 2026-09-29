import pytest
from arithmetic import *
from algebra import (
    UncertainGraph,
    CompleteGraph,
    Algebra,
    pgr,
    pgr_f,
    graph,
    gr,
    draw,
    draw_ascii,
    draw_ast,
)


class TestUncertainGraphLazyAST:
    def test_lazy_evaluation_trigger_on_print(self):
        """Tests that calculation does not occur until print/repr/str or to_set is invoked."""
        X = UncertainNumber({1, 2})
        f = lambda x: x + x
        g = pgr(f, X)

        # Before print, the graph must NOT be computed
        assert g.is_computed is False
        assert g._cached_points is None

        # Verify AST properties are present without computation
        assert g.ast["type"] == "pgr"
        assert g.ast["space_type"] == "minkowski"
        assert g.d == (2, 2, 2)

        # Calling str() / repr() (as done by print) triggers evaluation
        output_str = str(g)
        assert g.is_computed is True
        assert g._cached_points is not None
        assert output_str == "{(1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4)}_u"

    def test_example_3_6_minkowski(self):
        """
        Verification of Example 3.6 from the research monograph:
        For f = X + X, pgr_f({1, 2}_u) = {(1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4)}_u
        """
        X = UncertainNumber({1, 2})
        f = lambda x: x + x
        g = graph(f, X)

        expected_points = [(1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4)]
        assert g.points == expected_points
        assert g.domain_values == [1, 2]
        assert g.image_values == [2, 3, 4]
        assert repr(g) == "{(1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4)}_u"
    def test_example_3_6_mixed(self):
        X = UncertainNumber({1, 2})
        f1 = m(lambda x: x+x) 
        f2 = pw(lambda x: x+x)
        g = graph(lambda x:f1(x)+f2(x), X)
        assert g.points == [(1,4),(1,5), (1,6), (1,7),(1,8), (2,4),(2,5),(2,6),(2,7), (2,8)]
        assert g.domain_values == [1, 2]
        assert g.image_values == [4,5,6,7,8]
    def test_pointwise_graph(self):
        """Tests graph computation in Pointwise space (o)_1."""
        X = UncertainNumber({1, 2, 3, 4})
        f = lambda x: x**2
        g = pgr(f, X, space="pw")
        assert g.is_computed is False
        assert g.d == (4,)
        assert len(g) == 4
        assert g == UncertainNumber({(1,1),(2,4),(3,9),(4,16)})
        assert g.space_type == "pointwise"

        # Evaluate
        pts = g.points
        assert pts == [(1, 1), (2, 4), (3, 9), (4, 16)]
        assert repr(g) == "{(1, 1), (2, 4), (3, 9), (4, 16)}_u"
        assert g.domain_values == [1, 2, 3, 4]
        assert g.image_values == [1, 4, 9, 16]

    def test_complete_graph_gr(self):
        """Tests Complete Graph gr(f) := { pgr_f(X) : X in domain } (Definition 3.4)."""
        domain = [UncertainNumber({1, 2}), UncertainNumber({3, 4})]
        f = lambda x: x * 2
        cg = gr(f, domain, space="m")

        assert cg.is_computed is False
        assert len(cg) == 2
        assert cg.ast["type"] == "gr"

        all_pts = cg.all_points()
        assert cg.is_computed is True
        assert (1, 2) in all_pts
        assert (2, 4) in all_pts
        assert (3, 6) in all_pts
        assert (4, 8) in all_pts

    def test_uncertain_number_method_binding(self):
        """Tests that UncertainNumber.graph() and UncertainNumber.pgr() work via monkey-patch."""
        X = UncertainNumber({1, 2})
        f = lambda x: x + x
        g = X.graph(f)
        assert isinstance(g, UncertainGraph)
        assert g.points == [(1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4)]

    def test_algebra_class_api(self):
        """Tests the unified Algebra class static methods."""
        X = UncertainNumber({1, 2})
        f = lambda x: x + 1
        # In Minkowski space: A x f(A) = {1, 2} x {2, 3}
        g_m = Algebra.graph(f, X, space="m")
        assert g_m.points == [(1, 2), (1, 3), (2, 2), (2, 3)]

        # In Pointwise space: (x, x + 1)
        g_pw = Algebra.graph(f, X, space="pw")
        assert g_pw.points == [(1, 2), (2, 3)]


class TestDrawFunctions:
    def test_draw_return_fig(self):
        """Tests draw function returning matplotlib figure."""
        X = UncertainNumber({1, 2})
        g = pgr(lambda x: x + x, X)
        fig, ax = draw(g, return_fig=True, show=False)
        assert fig is not None
        assert ax is not None

    def test_draw_shorthand(self):
        """Tests draw(f, X) shorthand."""
        X = UncertainNumber({1, 2, 3})
        fig, ax = draw(lambda x: x**2, X, space="pw", return_fig=True, show=False)
        assert fig is not None

    def test_draw_ascii(self):
        """Tests ASCII terminal output."""
        X = UncertainNumber({1, 2})
        g = pgr(lambda x: x + x, X)
        ascii_art = draw_ascii(g)
        assert isinstance(ascii_art, str)
        assert "*" in ascii_art
        assert "y_max" in ascii_art
        assert "x_min" in ascii_art

    def test_draw_ast(self):
        """Tests AST hierarchy visualization."""
        X = UncertainNumber({1, 2})
        g = pgr(lambda x: x + x, X)
        ast_str = draw_ast(g, as_text=False)
        assert "AST Hierarchy:" in ast_str
        assert "[pgr]" in ast_str
