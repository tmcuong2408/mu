import itertools
from typing import Callable, Union, List, Tuple, Set, Any

# Relative imports within the 'arithmetic' package
from .UncertainNumber import UncertainNumber
from .PointwiseArithmetic import PointwiseArithmetic
from .EPointwiseArithmetic import EPointwiseArithmetic
from .MinkowskiArithmetic import MinkowskiArithmetic
from .EMinkowskiArithmetic import EMinkowskiArithmetic

# Type alias for numeric types
Numeric = Union[int, float, complex]

class Arithmetic:
    """
    Master Operator Overloading Engine.
    Executes operations across custom Arithmetic Spaces:
    - 'pw'  : Pointwise Arithmetic Space (o)_1
    - 'epw' : Extended Pointwise Arithmetic Space (o)_1'
    - 'm'   : Minkowski Arithmetic Space (o)_m (Default)
    - 'em'  : Extended Minkowski Arithmetic Space (o)_m'
    """

    @staticmethod
    def _ensure_uncertain(val: Any) -> UncertainNumber:
        """Helper to wrap scalar numeric values or collections into UncertainNumbers."""
        if isinstance(val, UncertainNumber):
            return val
        if isinstance(val, (set, list, tuple)):
            return UncertainNumber(val)
        return UncertainNumber({val})

    @classmethod
    def _dispatch(
        cls,
        a: Any,
        b: Any,
        operator_fn: Callable[[Numeric, Numeric], Numeric],
        space: str = "m",
        operator_symbol: str = "+",
    ) -> UncertainNumber:
        """
        Dispatches operation execution to the designated Arithmetic Space Engine.
        """
        a_unc = cls._ensure_uncertain(a)
        b_unc = cls._ensure_uncertain(b)
        space_key = space.lower()

        if space_key == "pw":
            return PointwiseArithmetic.pw(a_unc, b_unc, operator_fn, operator_symbol=operator_symbol)
        elif space_key == "epw":
            return EPointwiseArithmetic.epw(a_unc, b_unc, operator_fn, operator_symbol=operator_symbol)
        elif space_key == "m":
            return MinkowskiArithmetic.m(a_unc, b_unc, operator_fn, operator_symbol=operator_symbol)
        elif space_key == "em":
            return EMinkowskiArithmetic.em(a_unc, b_unc, operator_fn)
        else:
            raise ValueError(
                f"Unknown arithmetic space '{space}'. Expected one of: 'pw', 'epw', 'm', 'em'."
            )

    # ==================== PUBLIC ARITHMETIC METHODS ====================

    @classmethod
    def add(cls, a: Any, b: Any, space: str = "m") -> UncertainNumber:
        return cls._dispatch(a, b, lambda x, y: x + y, space=space, operator_symbol="+")

    @classmethod
    def sub(cls, a: Any, b: Any, space: str = "m") -> UncertainNumber:
        return cls._dispatch(a, b, lambda x, y: x - y, space=space, operator_symbol="-")

    @classmethod
    def mul(cls, a: Any, b: Any, space: str = "m") -> UncertainNumber:
        return cls._dispatch(a, b, lambda x, y: x * y, space=space, operator_symbol="*")

    @classmethod
    def truediv(cls, a: Any, b: Any, space: str = "m") -> UncertainNumber:
        return cls._dispatch(a, b, lambda x, y: x / y, space=space, operator_symbol="/")

    @classmethod
    def floordiv(cls, a: Any, b: Any, space: str = "m") -> UncertainNumber:
        return cls._dispatch(a, b, lambda x, y: x // y, space=space, operator_symbol="//")

    @classmethod
    def pow(cls, a: Any, b: Any, space: str = "m") -> UncertainNumber:
        return cls._dispatch(a, b, lambda x, y: x ** y, space=space, operator_symbol="**")

    @classmethod
    def mod(cls, a: Any, b: Any, space: str = "m") -> UncertainNumber:
        return cls._dispatch(a, b, lambda x, y: x % y, space=space, operator_symbol="%")

    @classmethod
    def neg(cls, a: Any, space: str = "m") -> UncertainNumber:
        return cls._dispatch(a, UncertainNumber({-1}), lambda x, y: x * y, space=space, operator_symbol="*")

    @classmethod
    def pos(cls, a: Any, space: str = "m") -> UncertainNumber:
        return cls._ensure_uncertain(a)

    @classmethod
    def abs(cls, a: Any, space: str = "m") -> UncertainNumber:
        unc = cls._ensure_uncertain(a)
        return UncertainNumber(
            generative_fn=lambda idx: abs(unc.evaluate_at_index(idx)),
            index_domain=unc.d,
            ast_node={"type": "custom_fn", "space_type": "minkowski"},
        )


# ==================== OVERLOAD DEFAULT MAGIC METHODS ====================
# Default magic operators (+, -, *, /, //, %, **) strictly fall back to Minkowski space 'm'

UncertainNumber.__add__ = lambda self, other: Arithmetic.add(self, other, space="m")
UncertainNumber.__radd__ = lambda self, other: Arithmetic.add(Arithmetic._ensure_uncertain(other), self, space="m")

UncertainNumber.__sub__ = lambda self, other: Arithmetic.sub(self, other, space="m")
UncertainNumber.__rsub__ = lambda self, other: Arithmetic.sub(Arithmetic._ensure_uncertain(other), self, space="m")

UncertainNumber.__mul__ = lambda self, other: Arithmetic.mul(self, other, space="m")
UncertainNumber.__rmul__ = lambda self, other: Arithmetic.mul(Arithmetic._ensure_uncertain(other), self, space="m")

UncertainNumber.__truediv__ = lambda self, other: Arithmetic.truediv(self, other, space="m")
UncertainNumber.__rtruediv__ = lambda self, other: Arithmetic.truediv(Arithmetic._ensure_uncertain(other), self, space="m")

UncertainNumber.__floordiv__ = lambda self, other: Arithmetic.floordiv(self, other, space="m")
UncertainNumber.__rfloordiv__ = lambda self, other: Arithmetic.floordiv(Arithmetic._ensure_uncertain(other), self, space="m")

UncertainNumber.__mod__ = lambda self, other: Arithmetic.mod(self, other, space="m")
UncertainNumber.__rmod__ = lambda self, other: Arithmetic.mod(Arithmetic._ensure_uncertain(other), self, space="m")

UncertainNumber.__pow__ = lambda self, other: Arithmetic.pow(self, other, space="m")
UncertainNumber.__rpow__ = lambda self, other: Arithmetic.pow(Arithmetic._ensure_uncertain(other), self, space="m")

UncertainNumber.__neg__ = lambda self: Arithmetic.neg(self, space="m")
UncertainNumber.__pos__ = lambda self: Arithmetic.pos(self, space="m")
UncertainNumber.__abs__ = lambda self: Arithmetic.abs(self, space="m")


# ==================== MINKOWSKI LIFTING FOR ARBITRARY SCALAR FUNCTIONS ====================
# General definition: for any scalar function f(a, b, ...) on real/complex numbers,
# lift to Minkowski space:
#   f_m(A, B, ...) = { f(a, b, ...) : a ∈ A, b ∈ B, ... }  (Cartesian product A × B × ...)

import builtins as _builtins

_builtin_max = _builtins.max
_builtin_min = _builtins.min


def _minkowski_lift(scalar_fn, *args, **kwargs):
    """
    Lift scalar function scalar_fn to Minkowski space (o)_m:
        f_m(A, B, ...) = { f(a, b, ...) : a ∈ A, b ∈ B, ... }
    The new index domain is the Cartesian product d_A × d_B × ...
    Generative function: f(i_1, i_2, ...) = scalar_fn(A[i_1], B[i_2], ...).
    """
    unc_args = [Arithmetic._ensure_uncertain(a) for a in args]
    new_d = sum((u.d for u in unc_args), ())

    def generative_fn(idx_tuple):
        if not isinstance(idx_tuple, tuple):
            idx_tuple = (idx_tuple,)
        vals = []
        curr = 0
        for u in unc_args:
            dim = len(u.d)
            sub_idx = idx_tuple[curr: curr + dim]
            curr += dim
            vals.append(u.evaluate_at_index(sub_idx))
        return scalar_fn(*vals)

    return UncertainNumber(
        generative_fn=generative_fn,
        index_domain=new_d,
        ast_node={"type": "custom_fn", "space_type": "minkowski"},
    )


def _unc_max(*args, **kwargs):
    """
    Override of max(), adhering to the binary operation definition on Minkowski space:
        max(A, B) = { max(a, b) : a ∈ A, b ∈ B }
    If no UncertainNumber is present in args, fallback to normal builtin max().
    """
    # key=/default= not supported in Minkowski lifting -> fallback to builtin
    if kwargs.get("key") is not None or kwargs.get("default") is not None:
        return _builtin_max(*args, **kwargs)
    # No UncertainNumber found -> fallback to builtin
    if not any(isinstance(a, UncertainNumber) for a in args):
        return _builtin_max(*args, **kwargs)
    # max(A) — single uncertain number: return maximum of its internal set
    if len(args) == 1 and isinstance(args[0], UncertainNumber):
        elems = list(args[0].to_set())
        return _builtin_max(elems)
    # max(A, B, ...) — multiple arguments: Minkowski lifting
    return _minkowski_lift(_builtin_max, *args)


def _unc_min(*args, **kwargs):
    """
    Override of min(), adhering to the binary operation definition on Minkowski space:
        min(A, B) = { min(a, b) : a ∈ A, b ∈ B }
    If no UncertainNumber is present in args, fallback to normal builtin min().
    """
    if kwargs.get("key") is not None or kwargs.get("default") is not None:
        return _builtin_min(*args, **kwargs)
    if not any(isinstance(a, UncertainNumber) for a in args):
        return _builtin_min(*args, **kwargs)
    # min(A) — single uncertain number: return minimum of its internal set
    if len(args) == 1 and isinstance(args[0], UncertainNumber):
        elems = list(args[0].to_set())
        return _builtin_min(elems)
    # min(A, B, ...) — multiple arguments: Minkowski lifting
    return _minkowski_lift(_builtin_min, *args)


def lift_m(scalar_fn, *args):
    """
    Lift scalar function scalar_fn to Minkowski space (o)_m.

    General definition: for scalar function f(a, b, ...) on real/complex numbers,
    extended to Minkowski space:
        lift_m(f, A, B, ...) = { f(a, b, ...) : a ∈ A, b ∈ B, ... }

    The new index domain is the Cartesian product d_A × d_B × ...
    Generative function: f(i_1, i_2, ...) = scalar_fn(A[i_1], B[i_2], ...).

    Examples:
        a = s(1, 2, 4, 6, 7, 8, 9, 20, 100)
        b = s(1, 2)
        lift_m(max, a, b)   # {max(x,y) : x∈a, y∈b} = {2, 4, 6, 7, 8, 9, 20, 100}_u
        lift_m(math.gcd, a, b)  # {gcd(x,y) : x∈a, y∈b}
    """
    return _minkowski_lift(scalar_fn, *args)


# Override builtins globally upon importing Arithmetic
_builtins.max = _unc_max
_builtins.min = _unc_min