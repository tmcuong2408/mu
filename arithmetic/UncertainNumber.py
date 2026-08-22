import math
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
            if isinstance(data, range):
                n = len(data)
                self.d: Tuple[int, ...] = (n,)
                self.elements = data
                if n == 0:
                    self.f = lambda x: None
                    self._formula_template = lambda v: "0"
                elif n == 1:
                    v = data[0]
                    self.f = lambda x: v
                    self._formula_template = lambda v: f"{v}"
                else:
                    start = data[0]
                    step = data.step
                    # Đơn thức bậc nhất O(1): f(x) = start + (x - 1) * step
                    self.f = lambda x: start + ((x[0] if isinstance(x, tuple) else x) - 1) * step
                    if step == 1 and start == 1:
                        self._formula_template = lambda v: f"{v}"
                    elif step == 1 and start == 0:
                        self._formula_template = lambda v: f"{v} - 1"
                    elif step == 1:
                        self._formula_template = lambda v: f"{start - 1} + {v}" if start > 1 else f"{v} - {1 - start}"
                    else:
                        self._formula_template = lambda v: f"{start} + ({v} - 1) * {step}"
                self.ast: dict = {"type": "leaf"}

            else:
                data_list = list(data)
                try:
                    y_values = sorted(data_list)
                except Exception:
                    y_values = data_list
                n = len(y_values)

                # 'd' is a tuple containing the maximum index size: d = (n,)
                self.d: Tuple[int, ...] = (n,)
                self.elements = y_values

                is_numeric = all(isinstance(v, (int, float, complex)) and not isinstance(v, bool) for v in y_values)
                if is_numeric and n > 0:
                    if n == 1:
                        v = y_values[0]
                        self.f = lambda x: v
                        self._formula_template = lambda v: f"{v}"
                    else:
                        # Kiểm tra cấp số cộng để dùng đơn thức bậc nhất f(x) = a*x + b (O(1))
                        step = (y_values[-1] - y_values[0]) / (n - 1)
                        is_arithmetic = False
                        if n <= 100:
                            is_arithmetic = all(abs(y_values[i] - (y_values[0] + i * step)) < 1e-9 for i in range(n))
                        else:
                            is_arithmetic = (
                                abs(y_values[1] - (y_values[0] + step)) < 1e-9 and
                                abs(y_values[n // 2] - (y_values[0] + (n // 2) * step)) < 1e-9 and
                                abs(y_values[-1] - (y_values[0] + (n - 1) * step)) < 1e-9
                            )

                        if is_arithmetic:
                            y0 = y_values[0]
                            # Đơn thức bậc nhất: f(x) = y0 + (x - 1) * step
                            self.f = lambda x: y0 + ((x[0] if isinstance(x, tuple) else x) - 1) * step
                            if step == 1 and y0 == 1:
                                self._formula_template = lambda v: f"{v}"
                            elif step == 1 and y0 == 0:
                                self._formula_template = lambda v: f"{v} - 1"
                            elif step == 1:
                                self._formula_template = lambda v: f"{y0 - 1} + {v}" if y0 > 1 else f"{v} - {1 - y0}"
                            else:
                                self._formula_template = lambda v: f"{y0} + ({v} - 1) * {step}"
                        elif n <= 10:
                            # Lagrange cho tập nhỏ
                            x_nodes = list(range(1, n + 1))
                            self.f = lagrange_interpolation(x_nodes, y_values)
                            self._formula_template = lambda v: f"LagrangePoly({v})"
                        else:
                            # Tra cứu mảng O(1) và nội suy tuyến tính từng đoạn
                            def _fast_eval(x: Numeric) -> Numeric:
                                idx_val = x[0] if isinstance(x, tuple) else x
                                if isinstance(idx_val, (int, float)) and 1 <= idx_val <= n:
                                    if isinstance(idx_val, int) or idx_val.is_integer():
                                        return y_values[int(idx_val) - 1]
                                    x_int = int(idx_val)
                                    if x_int == n:
                                        return y_values[-1]
                                    frac = idx_val - x_int
                                    return y_values[x_int - 1] * (1 - frac) + y_values[x_int] * frac
                                return None
                            self.f = _fast_eval
                            self._formula_template = lambda v: f"PiecewiseLinear({v})"
                else:
                    self.f = lambda idx: self.elements[int(idx[0] if isinstance(idx, tuple) else idx) - 1] if self.elements and 1 <= int(idx[0] if isinstance(idx, tuple) else idx) <= len(self.elements) else None
                    self._formula_template = lambda v: f"DiscreteLookup({v})"
                self.ast: dict = {"type": "leaf"}

        elif generative_fn is not None or index_domain is not None or ast_node is not None:
            # Pure Canonical Form initialization with index domain size tuple (f, d)
            self.d = tuple(index_domain) if index_domain is not None else ()
            self.ast = ast_node if ast_node is not None else {"type": "custom_fn"}
            self.f = generative_fn if generative_fn is not None else (lambda idx: self.evaluate_at_index(idx))
            self._formula_template = None

        else:
            raise ValueError(
                "Must provide either raw numeric data OR a generative function 'f' with index domain tuple 'd'."
            )

    def evaluate_at_index(self, index_key: Any) -> Any:
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

    def to_set(self) -> Set[Any]:
        """
        Executes Lazy Evaluation: Generates candidate index ranges (1..n_i) from tuple 'd'
        and evaluates connected AST nodes to output the final outcome set.
        """
        if self.d == (0,) or (self.ast.get("type") == "leaf" and hasattr(self, "elements") and not self.elements):
            return set()

        # Generate 1-based index ranges for each dimension represented in tuple 'd'
        index_ranges = [range(1, n + 1) for n in self.d]
        
        results = set()
        # Perform Cartesian product iteration over all index dimension ranges
        for idx in itertools.product(*index_ranges):
            try:
                val = self.evaluate_at_index(idx)
                if val is None:
                    continue
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

    def to_set_key(self) -> tuple:
        """Returns a hashable tuple representation of its set values."""
        try:
            s = self.to_set()
            formatted_elems = []
            for elem in s:
                if isinstance(elem, UncertainNumber):
                    formatted_elems.append(elem.to_set_key())
                else:
                    formatted_elems.append(elem)
            return tuple(sorted(formatted_elems, key=lambda x: str(x)))
        except Exception:
            return (id(self),)

    def __hash__(self):
        return hash(self.to_set_key())

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, UncertainNumber):
            return self.to_set_key() == other.to_set_key()
        return False

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, UncertainNumber):
            return str(self) < str(other)
        return False

    def flat_index_to_tuple(self, flat_idx: int, zero_based: bool = True) -> Tuple[int, ...]:
        """
        Converts a 1D linear integer index into a multi-dimensional scenario index tuple
        corresponding to index domain 'd' = (n_1, n_2, ..., n_k).
        Time complexity: O(k) (Pure Lazy Evaluation - avoids computing full Cartesian sets).
        """
        if not self.d or self.d == (0,):
            return ()

        total_size = math.prod(self.d)
        k0 = flat_idx if zero_based else (flat_idx - 1)

        # Support negative indexing (-1 = last element)
        if k0 < 0:
            k0 += total_size

        if not (0 <= k0 < total_size):
            raise IndexError(f"Index {flat_idx} out of range for UncertainNumber with {total_size} scenarios (domain {self.d}).")

        tuple_idx = []
        # Row-major / odometer decomposition from right to left
        for dim_size in reversed(self.d):
            if dim_size <= 0:
                tuple_idx.append(1)
            else:
                tuple_idx.append((k0 % dim_size) + 1)
                k0 //= dim_size

        return tuple(reversed(tuple_idx))

    def tuple_to_flat_index(self, tuple_idx: Tuple[int, ...], zero_based: bool = True) -> int:
        """
        Converts a multi-dimensional index tuple (1-based per dimension) into a 1D linear integer index.
        Time complexity: O(k).
        """
        if len(tuple_idx) != len(self.d):
            raise ValueError(f"Dimension mismatch: expected {len(self.d)} elements, got {len(tuple_idx)}")

        k0 = 0
        for idx_val, dim_size in zip(tuple_idx, self.d):
            if not (1 <= idx_val <= dim_size):
                raise IndexError(f"Index component {idx_val} out of bounds for dimension size {dim_size}")
            k0 = k0 * dim_size + (idx_val - 1)

        return k0 if zero_based else (k0 + 1)

    def __len__(self) -> int:
        """Returns the total number of scenarios N = prod(d)."""
        return math.prod(self.d) if self.d and self.d != (0,) else 0

    def __iter__(self):
        """Allows direct iteration over the evaluated set elements in sorted order."""
        return iter(sorted(list(self.to_set())))

    def __getitem__(self, index: Any):
        """
        Lazy O(k) scenario indexing:
        - If int: converts flat index -> multi-dimensional tuple index and evaluates f_X(tuple).
        - If tuple: evaluates directly at the given index tuple f_X(tuple).
        """
        if isinstance(index, tuple):
            return self.evaluate_at_index(index)
        elif isinstance(index, int):
            tuple_idx = self.flat_index_to_tuple(index, zero_based=True)
            val = self.evaluate_at_index(tuple_idx)
            if isinstance(val, float):
                val = round(val, 10)
                if val.is_integer():
                    val = int(val)
            return val
        elif isinstance(index, slice):
            total_size = len(self)
            start, stop, step = index.indices(total_size)
            return [self[i] for i in range(start, stop, step)]
        else:
            raise TypeError(f"UncertainNumber indices must be integers or tuples, not {type(index).__name__}")

    def get_formula(self, var_names: Union[List[str], Tuple[str, ...], None] = None) -> str:
        """
        Returns the symbolic mathematical formula f(x_1, x_2, ...) of the generative function.
        """
        num_vars = len(self.d) if self.d and self.d != (0,) else 1
        if var_names is None:
            if num_vars == 1:
                var_names = ["x"]
            else:
                var_names = [f"x_{i + 1}" for i in range(num_vars)]

        if self.ast.get("type") in ("leaf", "custom_fn"):
            if hasattr(self, "_formula_template") and self._formula_template is not None:
                v = var_names[0] if var_names else "x"
                return self._formula_template(v)
            elif hasattr(self, "elements") and len(self.elements) == 1:
                return str(self.elements[0])
            else:
                v_str = ", ".join(var_names) if var_names else "x"
                return f"f({v_str})"

        elif self.ast.get("type") == "op":
            left: "UncertainNumber" = self.ast["left"]
            right: "UncertainNumber" = self.ast["right"]
            op_sym = self.ast.get("operator_symbol", "+")
            space_type = self.ast.get("space_type", "minkowski")

            if space_type == "pointwise":
                left_vars = var_names
                right_vars = var_names
            else:
                left_dim = len(left.d) if left.d and left.d != (0,) else 1
                left_vars = var_names[:left_dim]
                right_vars = var_names[left_dim:]

            left_str = left.get_formula(left_vars)
            right_str = right.get_formula(right_vars)
            return f"({left_str}) {op_sym} ({right_str})"

        v_str = ", ".join(var_names) if var_names else "x"
        return f"f({v_str})"

    @property
    def formula(self) -> str:
        """Returns the canonical form formula representation: f(x_1, ...) = ..."""
        num_vars = len(self.d) if self.d and self.d != (0,) else 1
        if num_vars == 1:
            var_names = ["x"]
            head = "f(x)"
        else:
            var_names = [f"x_{i + 1}" for i in range(num_vars)]
            head = f"f({', '.join(var_names)})"
        return f"{head} = {self.get_formula(var_names)}"

    def __mul__(self, other: Any) -> "UncertainNumber":
        """Overloads '*' to act as an abstract AST composition operator."""
        return self.compose(other)

    def __repr__(self):
        try:
            s = self.to_set()
            if not s:
                return "{}_u"
            sorted_items = sorted(
                list(s),
                key=lambda x: (
                    1 if isinstance(x, UncertainNumber) else 0,
                    x.to_set_key() if isinstance(x, UncertainNumber) else x,
                ),
            )
            inner = ", ".join(repr(x) if isinstance(x, UncertainNumber) else str(x) for x in sorted_items)
            return f"{{{inner}}}_u"
        except Exception:
            return f"UncertainNumber(d={self.d})"

    def __str__(self):
        return self.__repr__()

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
    Evaluates algebraic operations across independent Minkowski scenarios.
    """
    if not args:
        return UncertainNumber({fn()})

    unc_args = [_to_unc(a) for a in args]

    # Try evaluating with UncertainNumber instances directly to leverage overloaded Minkowski operators
    try:
        res = fn(*unc_args)
        if isinstance(res, UncertainNumber):
            return res
        elif isinstance(res, (int, float, complex, set, list, tuple)):
            return _to_unc(res)
    except Exception:
        pass

    # Fallback to Cartesian scalar evaluation across all input scenarios
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

    Supports single-argument lambdas that encode scalar multiplication or
    power operations, e.g.:
        em(lambda A: 0.5 * A, A)   ->  (1/2 * A)_{m'}
        em(lambda A: A ** 0.5, A)  ->  (A^{1/2})_{m'}
    """
    if not args:
        return UncertainNumber({fn()})

    unc_args = [_to_unc(a) for a in args]

    # Check for 2-argument scalar operations suitable for EMinkowskiArithmetic
    if len(unc_args) == 2:
        from .EMinkowskiArithmetic import EMinkowskiArithmetic
        if unc_args[0].d == (1,) or unc_args[1].d == (1,):
            return EMinkowskiArithmetic.em(unc_args[0], unc_args[1], fn)

    # Handle single-argument lambdas of the form fn(A) = scalar * A or fn(A) = A ** scalar.
    # Probe fn with a scalar value to detect the hidden scalar operand.
    if len(unc_args) == 1:
        from .EMinkowskiArithmetic import EMinkowskiArithmetic
        unc_target = unc_args[0]
        probe = 4.0  # arbitrary non-trivial probe value

        # Detect scalar multiplication: fn(x) = c * x  =>  fn(probe)/probe == constant
        try:
            result_mul = fn(probe)
            if isinstance(result_mul, (int, float)) and probe != 0:
                scalar_val = result_mul / probe
                # Verify: fn(probe2) / probe2 == same constant
                probe2 = 9.0
                result_mul2 = fn(probe2)
                if isinstance(result_mul2, (int, float)) and abs(result_mul2 / probe2 - scalar_val) < 1e-9:
                    # Confirmed: fn is a scalar multiplication by scalar_val
                    scalar_unc = UncertainNumber({scalar_val})
                    return EMinkowskiArithmetic.em(scalar_unc, unc_target, lambda x, y: x * y)
        except Exception:
            pass

        # Detect power operation: fn(x) = x ** p  =>  log(fn(probe)) / log(probe) == p
        import math
        try:
            result_pow = fn(probe)
            if isinstance(result_pow, (int, float)) and probe > 0 and result_pow > 0:
                p_val = math.log(result_pow) / math.log(probe)
                probe2 = 9.0
                result_pow2 = fn(probe2)
                if isinstance(result_pow2, (int, float)) and result_pow2 > 0:
                    p_val2 = math.log(result_pow2) / math.log(probe2)
                    if abs(p_val2 - p_val) < 1e-9:
                        # Confirmed: fn is a power operation x ** p_val
                        power_unc = UncertainNumber({p_val})
                        return EMinkowskiArithmetic.em(unc_target, power_unc, lambda x, y: x ** y)
        except Exception:
            pass

    return m(fn, *unc_args)
