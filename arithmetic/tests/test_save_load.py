import os
import io
import json
import pytest
from fractions import Fraction
from decimal import Decimal
from arithmetic import (
    UncertainNumber,
    s,
    pw,
    m,
    save,
    load,
)
from arithmetic.PointwiseArithmetic import PointwiseArithmetic


class TestSaveLoadAST:
    def test_save_load_leaf_integers(self, tmp_path):
        """Tests saving and loading a simple leaf UncertainNumber of integers."""
        filepath = tmp_path / "leaf_int.json"
        X = s(1, 2, 3, 4)
        X.save(filepath)

        # File should be valid JSON
        assert os.path.exists(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["type"] == "leaf"
        assert "fn" in data
        assert "elements" not in data
        assert data["d"] == [4]

        # Load back
        X_loaded = UncertainNumber.load(filepath)
        assert isinstance(X_loaded, UncertainNumber)
        assert X_loaded.ast["type"] == "leaf"
        assert X_loaded.to_set() == {1, 2, 3, 4}
        assert X_loaded.d == (4,)

    def test_save_load_leaf_fractions_and_complex(self, tmp_path):
        """Tests saving and loading exact Fractions, Decimals, and Complex numbers."""
        filepath = tmp_path / "leaf_exact.json"
        X = UncertainNumber([Fraction(1, 3), Fraction(2, 5), complex(1, 2)])
        save(X, filepath)

        X_loaded = load(filepath)
        s_vals = X_loaded.to_set()
        assert Fraction(1, 3) in s_vals
        assert Fraction(2, 5) in s_vals
        assert complex(1, 2) in s_vals

    def test_save_load_leaf_range(self, tmp_path):
        """Tests saving and loading a range leaf."""
        filepath = tmp_path / "leaf_range.json"
        X = UncertainNumber(range(1, 10, 2))
        X.save(filepath)

        X_loaded = load(filepath)
        assert X_loaded.to_set() == {1, 3, 5, 7, 9}

    def test_save_load_ast_lazy_operation(self, tmp_path):
        """Tests saving and loading an AST tree without losing lazy evaluation."""
        filepath = tmp_path / "ast_op.json"
        A = s(1, 2)
        B = s(3, 4)
        C = A + B  # Minkowski sum: AST node with type='op'

        # Verify C is an op node
        assert C.ast["type"] == "op"
        assert C.ast["operator_symbol"] == "+"
        assert C.ast["space_type"] == "minkowski"
        assert C.d == (2, 2)

        # Save to file
        C.save(filepath)

        # Inspect raw JSON AST structure
        with open(filepath, "r", encoding="utf-8") as f:
            ast_json = json.load(f)
        assert ast_json["type"] == "op"
        assert ast_json["operator_symbol"] == "+"
        assert ast_json["space_type"] == "minkowski"
        assert ast_json["left"]["type"] == "leaf"
        assert "fn" in ast_json["left"]
        assert "elements" not in ast_json["left"]
        assert ast_json["right"]["type"] == "leaf"
        assert "fn" in ast_json["right"]
        assert "elements" not in ast_json["right"]

        # Load back
        C_loaded = load(filepath)
        assert C_loaded.ast["type"] == "op"
        assert C_loaded.ast["operator_symbol"] == "+"
        assert C_loaded.ast["space_type"] == "minkowski"
        assert C_loaded.d == (2, 2)

        # Index evaluation
        assert C_loaded.evaluate_at_index((1, 1)) == 4
        assert C_loaded.evaluate_at_index((1, 2)) == 5
        assert C_loaded.evaluate_at_index((2, 1)) == 5
        assert C_loaded.evaluate_at_index((2, 2)) == 6

        # Set evaluation
        assert C_loaded.to_set() == {4, 5, 6}

    def test_save_load_deep_nested_ast(self, tmp_path):
        """Tests deep nested expressions (A + B) * C - A."""
        filepath = tmp_path / "deep_ast.json"
        A = s(1, 2)
        B = s(3, 4)
        C = s(2, 5)

        expr = (A + B) * C - A
        save(expr, filepath)

        loaded_expr = load(filepath)
        assert loaded_expr.ast["type"] == "op"
        assert loaded_expr.ast["operator_symbol"] == "-"
        assert loaded_expr.to_set() == expr.to_set()
        assert loaded_expr.d == expr.d

    def test_save_load_pointwise_space(self, tmp_path):
        """Tests saving and loading operations in Pointwise space (o)_1."""
        filepath = tmp_path / "pointwise_ast.json"
        A = s(1, 2, 3)
        B = s(4, 5, 6)
        C = PointwiseArithmetic.pw(A, B, lambda x, y: x * y, operator_symbol="*")

        assert C.ast["space_type"] == "pointwise"
        assert C.d == (3,)
        C.save(filepath)

        C_loaded = load(filepath)
        assert C_loaded.ast["space_type"] == "pointwise"
        assert C_loaded.to_set() == {4, 10, 18}
        assert C_loaded.d == (3,)

    def test_save_load_weights(self, tmp_path):
        """Tests saving and loading UncertainNumbers with probability/frequency weights."""
        filepath = tmp_path / "weights.json"
        weights_dict = {1: Fraction(1, 4), 2: Fraction(3, 4)}
        X = UncertainNumber([1, 2], weights=weights_dict)
        X.save(filepath)

        X_loaded = load(filepath)
        assert X_loaded.to_set() == {1, 2}
        assert X_loaded.weights[1] == Fraction(1, 4)
        assert X_loaded.weights[2] == Fraction(3, 4)

    def test_save_load_pickle_format(self, tmp_path):
        """Tests saving and loading using pickle format."""
        filepath = tmp_path / "ast.pkl"
        A = s(1, 2)
        B = s(3, 4)
        C = A * B
        C.save(filepath, format="pickle")

        C_loaded = load(filepath, format="pickle")
        assert C_loaded.to_set() == C.to_set()

    def test_save_load_file_like_object(self):
        """Tests saving and loading via in-memory StringIO buffer."""
        A = s(10, 20)
        B = s(1, 2)
        C = A + B

        buf = io.StringIO()
        C.save(buf)
        buf.seek(0)

        C_loaded = UncertainNumber.load(buf)
        assert C_loaded.to_set() == C.to_set()

    def test_to_json_from_json(self):
        """Tests to_json() and from_json() direct string methods."""
        X = s(5, 10)
        Y = s(1, 2)
        Z = X * Y

        json_str = Z.to_json()
        assert isinstance(json_str, str)
        assert '"type": "op"' in json_str

        Z_loaded = UncertainNumber.from_json(json_str)
        assert Z_loaded.to_set() == {5, 10, 20}

    def test_save_auto_creates_subdirectories(self, tmp_path):
        """Tests that save() automatically creates parent directories if needed."""
        filepath = tmp_path / "deep" / "sub" / "folder" / "num.json"
        X = s(1, 2, 3)
        save(X, filepath)
        assert os.path.exists(filepath)
        assert load(filepath).to_set() == {1, 2, 3}

    def test_save_parameter_order_interchangeable(self, tmp_path):
        """Tests that module-level save(unc, path) and save(path, unc) both work."""
        p1 = tmp_path / "p1.json"
        p2 = tmp_path / "p2.json"
        X = s(10, 20)

        save(X, p1)
        save(p2, X)

        assert load(p1).to_set() == {10, 20}
        assert load(p2).to_set() == {10, 20}

    def test_save_load_generative_leaf(self, tmp_path):
        """Tests saving and loading a leaf node created via generative_fn."""
        filepath = tmp_path / "gen_leaf.json"
        X = UncertainNumber(
            generative_fn=lambda idx: idx * 2 if not isinstance(idx, tuple) else idx[0] * 2,
            index_domain=(4,),
            ast_node={"type": "leaf"},
        )
        save(X, filepath)

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["type"] == "leaf"
        assert "fn" in data
        assert "elements" not in data

        loaded = load(filepath)
        assert loaded.to_set() == {2, 4, 6, 8}

    def test_save_load_large_trillion_scale(self, tmp_path):
        """Tests saving and loading 10^9 scenarios without storing elements or expanding RAM."""
        filepath = tmp_path / "billion.json"
        A = UncertainNumber(range(1, 1_000_000_001))
        save(A, filepath)

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["type"] == "leaf"
        assert "fn" in data
        assert "elements" not in data
        assert data["fn"]["type"] == "poly"
        assert data["d"] == [1_000_000_000]

        # Loading must be instantaneous and O(1) RAM without expanding 1 billion elements
        loaded = load(filepath)
        assert loaded.d == (1_000_000_000,)
        assert loaded.evaluate_at_index((1,)) == 1
        assert loaded.evaluate_at_index((500_000_000,)) == 500_000_000
        assert loaded.evaluate_at_index((1_000_000_000,)) == 1_000_000_000

    def test_save_load_quintillion_scale(self, tmp_path):
        """Tests saving and loading 10^18 scenarios (Minkowski sum of two 10^9 numbers)."""
        filepath = tmp_path / "quintillion.json"
        A = UncertainNumber(range(1, 1_000_000_001))
        B = UncertainNumber(range(100, 100_000_000_100, 100))
        C = A + B

        assert len(C) == 1_000_000_000_000_000_000
        save(C, filepath)

        # File size on disk should be < 1 KB (compact function descriptors)
        file_size = os.path.getsize(filepath)
        assert file_size < 2048

        loaded = load(filepath)
        assert len(loaded) == 1_000_000_000_000_000_000
        assert loaded.ast["type"] == "op"
        assert loaded[0] == 101
        assert loaded[len(loaded) - 1] == 101_000_000_000
