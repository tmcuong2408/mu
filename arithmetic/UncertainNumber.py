import itertools
from typing import Callable, Union, List, Tuple, Set, Any

# Type alias for numeric types (supporting both Real and Complex numbers)
Numeric = Union[int, float, complex]


def lagrange_interpolation(x_nodes: List[int], y_values: List[Numeric]) -> Callable[[Numeric], Numeric]:
    """
    Constructs a generative function f(x) via Lagrange Polynomial Interpolation[cite: 1].
    Maps index nodes {1, ..., n} to corresponding numeric target values[cite: 1].
    """
    n = len(x_nodes)

    def generative_fn(x: Numeric) -> Numeric:
        total = 0
        for i in range(n):
            xi, yi = x_nodes[i], y_values[i]
            li = 1
            for j in range(n):
                if i != j:
                    xj = x_nodes[j]
                    li *= (x - xj) / (xi - xj)
            total += yi * li
        return total

    return generative_fn


class UncertainNumber:
    """
    Represents an Uncertain Number U(K) defined strictly via its Canonical Form (f_X, d_X).
    The index domain 'd' is a Tuple of integers where each element represents 
    the maximum index size (n_i) for that dimension.
    Joined via Abstract AST Composition '*'.
    """

    def __init__(
        self,
        data: Union[List[Numeric], Set[Numeric], None] = None,
        generative_fn: Union[Callable, None] = None,
        index_domain: Union[Tuple[int, ...], None] = None,
        ast_node: Union[dict, None] = None,
    ):
        """
        Initializer for an Uncertain Number:
        1. From raw numeric data: Constructs Lagrange polynomial generative function f(x).
        2. From pure Canonical Form: Generative function 'f' and index domain tuple 'd' representing sizes.
        """
        if data is not None:
            # Sort input values to establish a deterministic 1-based index mapping
            y_values = sorted(list(data))
            n = len(y_values)

            # 'd' is a tuple containing the maximum index size: d = (n,)
            self.d: Tuple[int, ...] = (n,)

            # Construct pure generative function f_X via Lagrange Interpolation
            x_nodes = list(range(1, n + 1))
            self.f: Callable[[Numeric], Numeric] = lagrange_interpolation(x_nodes, y_values)
            self.ast: dict = {"type": "leaf"}

        elif generative_fn is not None or index_domain is not None or ast_node is not None:
            # Pure Canonical Form initialization with index domain size tuple (f, d)
            self.d = tuple(index_domain) if index_domain is not None else ()
            self.ast = ast_node if ast_node is not None else {"type": "custom_fn"}
            self.f = generative_fn if generative_fn is not None else (lambda idx: self.evaluate_at_index(idx))

        else:
            raise ValueError(
                "Must provide either raw numeric data OR a generative function 'f' with index domain tuple 'd'."
            )

    def evaluate_at_index(self, index_key: Any) -> Numeric:
        """
        Evaluates the generative function f_X at a specific index key or index tuple.
        Recursively traverses AST node connections.
        """
        if not isinstance(index_key, tuple):
            index_key = (index_key,)

        if self.ast.get("type") in ("leaf", "custom_fn"):
            idx = index_key[0] if len(index_key) == 1 else index_key
            return self.f(idx)

        elif self.ast.get("type") == "op":
            left: "UncertainNumber" = self.ast["left"]
            right: "UncertainNumber" = self.ast["right"]
            operator_fn: Callable = self.ast["operator_fn"]
            space_type = self.ast.get("space_type", "minkowski")

            if space_type == "pointwise":
                if left.d == right.d:
                    idx_left = index_key
                    idx_right = index_key
                elif left.d == (1,):
                    idx_left = (1,)
                    idx_right = index_key
                elif right.d == (1,):
                    idx_left = index_key
                    idx_right = (1,)
                else:
                    idx_left = index_key
                    idx_right = index_key
            else:  # minkowski
                left_dim = len(left.d)
                idx_left = index_key[:left_dim]
                idx_right = index_key[left_dim:]

            val_left = left.evaluate_at_index(idx_left)
            val_right = right.evaluate_at_index(idx_right)

            # Dynamically execute abstract operator function
            return operator_fn(val_left, val_right)

        return self.f(index_key[0] if len(index_key) == 1 else index_key)

    def to_set(self) -> Set[Numeric]:
        """
        Executes Lazy Evaluation: Generates candidate index ranges (1..n_i) from tuple 'd'
        and evaluates connected AST nodes to output the final outcome set.
        """
        # Generate 1-based index ranges for each dimension represented in tuple 'd'
        index_ranges = [range(1, n + 1) for n in self.d]
        
        results = set()
        # Perform Cartesian product iteration over all index dimension ranges
        for idx in itertools.product(*index_ranges):
            try:
                val = self.evaluate_at_index(idx)
                if isinstance(val, float):
                    val = round(val, 10)
                    if val.is_integer():
                        val = int(val)
                elif isinstance(val, complex):
                    real = round(val.real, 10)
                    imag = round(val.imag, 10)
                    if real.is_integer():
                        real = int(real)
                    if imag.is_integer():
                        imag = int(imag)
                    val = complex(real, imag)
                results.add(val)
            except ZeroDivisionError:
                continue
        return results

    # ==================== ABSTRACT AST COMPOSITION OPERATOR '*' ====================

    def compose(self, other: Any, operator_fn: Union[Callable, None] = None) -> "UncertainNumber":
        """
        Abstract composition method: Joins two AST trees under a custom operator function.
        Concatenates index domain tuples d_A and d_B to form combined dimension sizes.
        """
        if not isinstance(other, UncertainNumber):
            other = UncertainNumber([other])

        if operator_fn is None:
            operator_fn = lambda a, b: a * b

        # Concatenate dimension size tuples: d_new = d_A + d_B
        new_d = self.d + other.d

        ast_node = {
            "type": "op",
            "left": self,
            "right": other,
            "operator_fn": operator_fn,
            "space_type": "minkowski",
        }

        # Return new UncertainNumber holding combined AST root and dimension tuple
        res = UncertainNumber(
            generative_fn=lambda idx: res.evaluate_at_index(idx),
            index_domain=new_d,
            ast_node=ast_node,
        )
        return res

    def __mul__(self, other: Any) -> "UncertainNumber":
        """Overloads '*' to act as an abstract AST composition operator."""
        return self.compose(other)

    def __repr__(self):
        return f"UncertainNumber(d={self.d}, ast_type='{self.ast['type']}')"

    # ==================== FUNCTIONAL SPACE OPERATORS (LAMBDAS) ====================

    @staticmethod
    def pw(fn: Callable, *args: Any) -> "UncertainNumber":
        """
        Executes a lambda function across arguments in Point-wise Space (o)_1.
        Preserves internal variable identity: all occurrences share the identical point value.
        Example: UncertainNumber.pw(lambda x: x**2 - 3*x + 2, X)
        """
        return pw(fn, *args)

    @staticmethod
    def epw(fn: Callable, *args: Any) -> "UncertainNumber":
        """
        Executes a lambda function across arguments in Extended Point-wise Space (o)_1'.
        Allows broadcasting with scalar values or (1,) domain dimensions.
        Example: UncertainNumber.epw(lambda x, c: x * c, X, 10)
        """
        return epw(fn, *args)

    @staticmethod
    def m(fn: Callable, *args: Any) -> "UncertainNumber":
        """
        Executes a lambda function across arguments in Minkowski Space (o)_m.
        Constructs Cartesian product domain d_1 x d_2 x ... for independent interaction.
        Example: UncertainNumber.m(lambda a, b: a + b, A, B)
        """
        return m(fn, *args)

    @staticmethod
    def em(fn: Callable, *args: Any) -> "UncertainNumber":
        """
        Executes a lambda function across arguments in Extended Minkowski Space (o)_m'.
        Supports inverse equation solving and fractional powers.
        Example: UncertainNumber.em(lambda c, x: c * x, 10, X)
        """
        return em(fn, *args)


# ==================== MODULE-LEVEL CONVENIENCE FUNCTIONS ====================

def _to_unc(val: Any) -> UncertainNumber:
    if isinstance(val, UncertainNumber):
        return val
    if isinstance(val, (set, list, tuple)):
        return UncertainNumber(val)
    return UncertainNumber({val})


def pw(fn: Callable, *args: Any) -> UncertainNumber:
    """
    Point-wise Space (o)_1 functional operator for lambdas with UncertainNumber parameters.
    Definition 3.2: (f)_1(A) := {f(x) : x in A}_u
    """
    if not args:
        return UncertainNumber({fn()})

    unc_args = [_to_unc(a) for a in args]
    d0 = unc_args[0].d

    for u in unc_args:
        if u.d != d0:
            raise ValueError(
                f"Point-wise Space (o)_1 requires identical index domains. Got {u.d} and {d0}."
            )

    generative_fn = lambda idx: fn(*(u.evaluate_at_index(idx) for u in unc_args))
    return UncertainNumber(
        generative_fn=generative_fn,
        index_domain=d0,
        ast_node={"type": "custom_fn", "space_type": "pointwise"},
    )


def epw(fn: Callable, *args: Any) -> UncertainNumber:
    """
    Extended Point-wise Space (o)_1' functional operator for lambdas.
    Supports broadcasting between scalars and multi-element UncertainNumbers.
    """
    if not args:
        return UncertainNumber({fn()})

    unc_args = [_to_unc(a) for a in args]
    non_scalar_domains = [u.d for u in unc_args if u.d != (1,)]

    if not non_scalar_domains:
        target_d = (1,)
    else:
        target_d = non_scalar_domains[0]
        for d in non_scalar_domains:
            if d != target_d:
                raise ValueError(
                    f"Incompatible multi-element domains for (o)_1': {d} and {target_d}."
                )

    generative_fn = lambda idx: fn(
        *(u.evaluate_at_index((1,) if u.d == (1,) else idx) for u in unc_args)
    )
    return UncertainNumber(
        generative_fn=generative_fn,
        index_domain=target_d,
        ast_node={"type": "custom_fn", "space_type": "pointwise"},
    )


def m(fn: Callable, *args: Any) -> UncertainNumber:
    """
    Minkowski Space (o)_m functional operator for lambdas with UncertainNumber parameters.
    Evaluates independent Cartesian scenarios across all inputs.
    """
    if not args:
        return UncertainNumber({fn()})

    unc_args = [_to_unc(a) for a in args]
    new_d = sum((u.d for u in unc_args), ())

    def generative_fn(idx_tuple: Any) -> Numeric:
        if not isinstance(idx_tuple, tuple):
            idx_tuple = (idx_tuple,)
        arg_vals = []
        curr = 0
        for u in unc_args:
            dim = len(u.d)
            sub_idx = idx_tuple[curr : curr + dim]
            curr += dim
            arg_vals.append(u.evaluate_at_index(sub_idx))
        return fn(*arg_vals)

    return UncertainNumber(
        generative_fn=generative_fn,
        index_domain=new_d,
        ast_node={"type": "custom_fn", "space_type": "minkowski"},
    )


def em(fn: Callable, *args: Any) -> UncertainNumber:
    """
    Extended Minkowski Space (o)_m' functional operator for lambdas.
    Dispatches to Extended Minkowski equation solving engine for scalars,
    or falls back to Cartesian Minkowski evaluation.
    """
    if not args:
        return UncertainNumber({fn()})

    unc_args = [_to_unc(a) for a in args]

    # Check for 2-argument scalar operations suitable for EMinkowskiArithmetic
    if len(unc_args) == 2:
        from .EMinkowskiArithmetic import EMinkowskiArithmetic
        if unc_args[0].d == (1,) or unc_args[1].d == (1,):
            return EMinkowskiArithmetic.em(unc_args[0], unc_args[1], fn)

    return m(fn, *unc_args)
