# Algebra
Mathematics of Uncertainty: Algebraic structures, uncertain functions, and graphs.

Trần, M. C. (2026). Extended logic and mathematics of uncertainty. <br/>
https://doi.org/10.5281/zenodo.22031881

## Author & Contact

**Trần Mạnh Cường**  
*Alumnus, Faculty of Mathematics & Informatics (K11)*  
*Thai Nguyen University of Sciences (TNUS), Vietnam*

* **Research Focus:** Mathematics of Uncertainty
* **Email:** tmcuong2408@gmail.com
* **Phone:** (+84) 353-237-140
* **Location:** DJ7 Street, Thoi Hoa, Ho Chi Minh City, Vietnam

---

## Theory: Graphs of Uncertain Functions

Based on **Chapter 3: Uncertain Algebra** in *Extended Logic and Mathematics of Uncertainty*:

### 1. Point-wise Graph of an Uncertain Function (Definition 3.3)
Let $f : \mathcal{X} \to \mathcal{Y}$ be an uncertain function, $A \in \mathcal{X}$. The graph of $f$ at $A$ is defined as:
$$pgr_f(A) := A \times f(A)$$

* **Example 3.6:** For $f = X + X$ and $A = \{1, 2\}_u$, in Minkowski arithmetic space:
  $$f(A) = \{1, 2\}_u + \{1, 2\}_u = \{2, 3, 4\}_u$$
  $$pgr_f(\{1, 2\}_u) = \{(1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4)\}_u$$

* In **Pointwise Arithmetic Space $(o)_1$**, internal point identity is preserved ($x \mapsto f(x)$):
  $$pgr_f(A) = \{(x, f(x)) : x \in A\}_u$$

### 2. Complete Graph of an Uncertain Function (Definition 3.4)
Let $\mathcal{X}, \mathcal{Y} \subseteq U(\mathbb{K})$. The complete graph over domain $\mathcal{X}$ is:
$$gr(f) := \{ pgr_f(X) : X \in \mathcal{X} \}$$

### 3. AST Structure & Lazy Evaluation
- When calling `graph(f, A)` or `pgr(f, A)`, the graph is represented as an Abstract Syntax Tree (AST) node connecting the scenario index domains:
  - Minkowski Space: $d_{pgr} = d_A \times d_{f(A)}$
  - Pointwise Space: $d_{pgr} = d_A$
- **Strictly Lazy**: No scenario coordinate pairs are evaluated upon creation.
- **Evaluation on Demand**: Actual calculation only takes place when `print(g)`, `repr(g)`, `g.to_set()`, or `draw(g)` is called.

---

## Usage

```python
from arithmetic import UncertainNumber
from algebra import graph, pgr, gr, draw, draw_ascii, draw_ast, Algebra

# 1. Graph in Minkowski space (o)_m (Example 3.6 from the monograph)
X = UncertainNumber({1, 2})
f = lambda x: x + x

g = pgr(f, X)
print(g.is_computed)   # False (Only the AST tree is constructed, no calculations yet)

# Calculations execute strictly when printed:
print(g)               # {(1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4)}_u
print(g.is_computed)   # True (Evaluated and cached)

# 2. Graph in Point-wise Space (o)_1
f_pw = lambda x: x**2
g_pw = pgr(f_pw, UncertainNumber({1, 2, 3, 4}), space="pw")
print(g_pw)            # {(1, 1), (2, 4), (3, 9), (4, 16)}_u

# 3. Complete Graph over a scenario domain (Definition 3.4)
domain = [UncertainNumber({1, 2}), UncertainNumber({3, 4})]
cg = gr(lambda x: x * 2, domain)
print(cg)              # {{(1, 2), (1, 4), (2, 2), (2, 4)}_u, {(3, 6), (3, 8), (4, 6), (4, 8)}_u}_u

# 4. Drawing the calculated graph
# Using matplotlib with automatic headless safety and terminal ASCII fallback:
draw(g, title="Graph of f(X) = X + X")

# Save to an image file:
draw(g, save_path="graph_plot.png")

# Or render directly via object method:
g.draw()

# 5. Visualizing in pure ASCII (for terminal environments)
print(draw_ascii(g))

# 6. Inspecting the AST Hierarchy
draw_ast(g)
```
