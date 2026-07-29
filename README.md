# nabla

**Symbolic automatic differentiation over [moissanite](https://github.com/minamorl/moissanite)
expression DAGs. Reverse mode with no runtime tape.**

A moissanite kernel is an inspectable expression tree assembled at runtime. Differentiation is a
transformation of that tree, so a derivative is _also_ a tree — and it goes down the same path to C
and `-O3`. The calculus happens while you build the kernel; **at run time there is nothing left to
do but arithmetic.**

```ruby
require 'nabla'

grad = Moissanite.kernel(:grad, g: :f64_buf, w: :f64_buf, x: :f64_buf, n: :i64) do |k, g, w, x, n|
  prediction = (0...8).inject(Moissanite::Expr.lift(0.0, :f64)) { |s, j| s + (w[j] * x[j]) }
  loss = (prediction - 1.0) * (prediction - 1.0)

  Nabla.emit_gradient(k, loss, (0...8).map { |j| w[j] })
       .each_with_index { |partial, j| k.store(g, j, partial) }
  k.ret 0
end
```

## Why there is no tape

Ordinary reverse-mode AD records operations as it executes them, then walks that tape backwards.
The tape costs memory and branches at run time, and it is the reason autodiff frameworks are
frameworks.

nabla needs none of it, and the reason is structural: **the program is data, and it is static.**
The whole expression is known the moment it is built, so the reverse walk is a DAG traversal at
_kernel-build time_. There is no execution to record.

That is not a micro-optimisation. It is what makes every partial derivative fall out of **one**
traversal:

| inputs | forward mode (nodes) | reverse mode (nodes) |  ratio |
| -----: | -------------------: | -------------------: | -----: |
|      1 |                   23 |                   18 |  1.28× |
|      5 |                  300 |                   50 |  6.00× |
|     10 |                1,050 |                   90 | 11.67× |
|     20 |                3,900 |                  170 | 22.94× |
|     50 |               23,250 |                  410 | 56.71× |

Forward mode differentiates once per input, so its total grows quadratically. Reverse mode grows
linearly — about 8 nodes per parameter. Reproduce with `rake examples`.

The same holds in wall-clock. On a least-squares loss over 200,000 samples, the **gradient for
every parameter costs ~1.2–1.6× the loss itself, and that ratio does not grow with the parameter
count** (measured from 4 to 64 parameters).

## Correctness

nabla inherits moissanite's constitution: **the pure-Ruby oracle is the semantics.** A derivative
kernel is a kernel, so the same differential testing applies — `test/equivalence_test.rb` pins
oracle and native to bit-for-bit agreement, including for third derivatives and for gradients taken
through a reduction loop.

Against hand-written analytic solutions, the least-squares gradient lands at **3.9e-16 relative
error** (a couple of ulp), and derivatives through `k.let` bindings match analytically **exactly**.

Forward and reverse mode agree to within a few ulp of each other. They are not bit-identical, and
that is not a defect: they accumulate adjoints in different association orders, and floating-point
addition is not associative. Each mode _is_ bit-identical to its own oracle, which is the invariant
that actually matters.

## A zero gradient is a claim, not a default

The most dangerous failure in autodiff is a silently wrong zero. `d/dx = 0` asserts "this input
does not affect the result" — turning _not knowing_ into an assertion is a bug nobody catches,
because nothing crashes and the number looks plausible.

So nabla never returns a zero it cannot justify. A `k.let` binding whose definition was not
supplied raises instead:

```
Nabla::Opaque: cannot differentiate through `v0`: it is a k.let binding whose definition was
not supplied. Pass env: Nabla.local_env(builder.statements), or build the primal as a plain
expression and let Nabla::Emitter reintroduce the bindings.
```

An input that genuinely does not reach the output still gets an exact zero — that one is a fact,
not a guess.

## Differentiating through `k.let`

Kernel bodies bind intermediates with `k.let`, so expressions contain `Local` nodes whose
definitions live in the statement list rather than the tree. Two ways through:

```ruby
# 1. hand the definitions to the differentiator
env = Nabla.local_env(builder.statements)
Nabla.gradient(expression, targets, env: env)

# 2. or fold a straight-line body back into one expression
Nabla.gradient(Nabla.flatten(builder.statements), targets)
```

Folding does not make the expression bigger, because the result is a DAG: a `let` becomes a shared
node rather than a copy, and `Nabla::Emitter` turns it back into a `let` on the way out. It is a
round trip.

**Locals that are reassigned with `k.assign` are deliberately excluded** from the environment. They
are loop-carried state, not pure bindings; inlining their initial value would quietly differentiate
a different function.

## Reductions

For a **sum** reduction, the adjoint of every iteration's contribution is exactly 1 — so the
gradient of the loss is the sum of the gradients of the terms. Apply reverse mode to the loop body
and accumulate. The loop still never needs recording:

```ruby
accumulators = (0...D).map { k.let(0.0) }
k.count(n) do |i|
  residual = prediction(w, x, k.let(i * D)) - y[i]   # plain expression, not k.let
  Nabla.emit_gradient(k, residual * residual, (0...D).map { |j| w[j] })
       .each_with_index { |partial, j| k.assign(accumulators[j], accumulators[j] + partial) }
end
```

Sequential recurrences — where iteration `i` depends on `i-1` — are **not** covered by this and
need a real tape or a statement-level transform. Sum- and product-shaped reductions are.

## Emitter is not an optimisation

Reverse mode produces heavily shared structure by construction: every partial references the same
intermediates. moissanite's expression type is a _tree_, so emitting without preserving that
sharing expands the DAG back out — and the O(N) you just won becomes O(N²) again.

At 50 inputs the gradient DAG is 410 nodes and **41,658 as a tree**. `Nabla::Emitter` binds every
subtree referenced more than once to a `k.let`, which is why that number never reaches the
compiler. Use `Nabla.emit_gradient` / `Nabla.emit` and it is handled.

## High-order derivatives

`Nabla.derivative` returns an expression, so it composes:

```ruby
third = 3.times.inject(potential) { |acc, _| Nabla.derivative(acc, x) }
```

Node counts grow roughly 5× per order (42 → 174 → 838 → 4,642), but gcc's CSE absorbs most of it:
a third derivative with 20× the nodes of the first runs at **1.69×** its cost. Compile time is what
degrades first — about 185 ms and 143 KB of C at the fifth order.

## The symbolic layer: integrals, multiple integrals, limits

Differentiation is compositional — every node has a rule, so the transformation always
terminates. **Integration is not.** There is no chain rule for `∫`, deciding elementary
integrability needs the Risch algorithm, and no CAS implements it completely.

So this layer solves the forms it can and **refuses the rest out loud**:

```ruby
Nabla.integrate((3.0 * x * x) + (2.0 * x) + 1.0, x)   # => x^3 + x^2 + x
Nabla.integrate(((2.0 * x) + 1.0).sin, x)             # => -0.5 * cos(2x + 1)
Nabla.integrate(x.sin * x.cos, x)
# Nabla::NotElementary: product of two factors that both depend on the variable
#                       needs integration by parts
```

Returning an approximation there would be the worst outcome: the caller cannot tell it apart
from a correct answer. Covered today: polynomials, linearity, `sin`/`cos`/`exp` of a linear
argument, `1/(ax+b)`, and constants. Integration by parts, partial fractions and general
substitution are not implemented.

### ∫ checked by ∂

This is why the layer lives next to a differentiator. `∫f = F` is exactly the claim `F' = f`,
and **nabla can check its own answer**:

```ruby
primitive = Nabla.integrate(integrand, x)
Nabla.verify_integral(integrand, primitive, x)
# => symbolically exact
# => numeric agreement, max relative error 3.49e-16
```

The two statuses are not the same kind of thing, and the API keeps them apart:

- **`:symbolic`** — `F'` reduced to `f` syntactically. No numbers were involved, so the integral
  is correct as long as the differentiation rules are. This is a certificate.
- **`:numeric`** — they did not match syntactically, so both were compiled to native kernels and
  compared over random points. This only says no counterexample was found.

Currently **2 of 8** worked examples verify symbolically; the rest fall back to numeric agreement
at 1e-14 or better. That ratio is a function of how strong the simplifier is, which makes it a
development metric rather than a slogan — run `rake examples` to see it.

To be clear about the category: this is a CAS, not a proof assistant. It computes; it does not
check proofs against a trusted kernel, and it is not comparable to Lean or Coq.

### Multiple integrals need no special machinery

A definite integral returns an _expression_, so the inner result is simply the outer integrand:

```ruby
inner = Nabla.definite_integral((x * x) + (y * y), x, 0.0, 1.0)   # an expression in y
outer = Nabla.definite_integral(inner, y, 0.0, 1.0)               # => 2/3
```

Bounds may be symbolic too, so `∫_0^y 2x dx = y²` works.

### Limits

L'Hopital costs almost nothing here, because the differentiator already exists:

```ruby
Nabla.limit(x.sin, x, x, at: 0.0)                    # => 1.0
Nabla.limit(1.0 - x.cos, x * x, x, at: 0.0)          # => 0.5
```

**This is the weakest part of the library.** Detecting the indeterminate form is a _numeric_
test against a threshold, not a symbolic one, so a very small non-zero value can be mistaken for
zero. Series-based limits (Gruntz) are not implemented; forms L'Hopital cannot resolve are
refused.

### Simplification

`Nabla.simplify` does constant folding and identity elimination. It is **not** an algebraic
normal form — it folds `x + x` but not `2x + 3x`, and it does not expand or collect like terms.

It matters more than it sounds: differentiation is syntactic, so it always leaves `1.0 * x + x *
1.0` behind. Simplifying shrinks derivative expressions by about 40% (`x*x` → 5 nodes → 3), and
without it the integrator's pattern matching cannot recognise `x * 1.0` as a polynomial.

Two things are deliberately left unfolded: division by zero (moissanite's oracle raises there and
native is undefined — the simplifier must not manufacture an `Inf`) and out-of-domain `log`/`sqrt`.

## Install

```ruby
gem 'nabla'
```

Requires Ruby 3.2+ and moissanite. A C toolchain is optional: without one, derivative kernels fall
back to the pure-Ruby oracle and still produce identical results, just slowly — the same fallback
chain moissanite guarantees.

## API

| Method                                              | Description                                                                     |
| --------------------------------------------------- | ------------------------------------------------------------------------------- |
| `Nabla.gradient(expr, targets, env: nil)`           | Every partial from one reverse traversal. Returns `{ target => Expr }`.         |
| `Nabla.derivative(expr, target, env: nil)`          | Forward mode, one input. Returns an expression, so it composes for high orders. |
| `Nabla.emit_gradient(builder, expr, targets, env:)` | Gradient + shared emission in one step. Returns values in target order.         |
| `Nabla.emit(builder, expr, env: nil)`               | Emit one expression, preserving sharing.                                        |
| `Nabla.local_env(statements)`                       | `{ Local => defining expression }`, excluding reassigned locals.                |
| `Nabla.flatten(statements)`                         | Fold a straight-line body (`Let`/`Ret`) into one expression.                    |
| `Nabla.size(expr)` / `Nabla.tree_size(expr)`        | DAG size vs. tree expansion. The gap is why `Emitter` exists.                   |

| `Nabla.simplify(expr)` | Constant folding and identity elimination. Not a normal form. |
| `Nabla.integrate(expr, var)` | Antiderivative, or `NotElementary` — never an approximation. |
| `Nabla.definite_integral(expr, var, lo, hi)` | `F(hi) - F(lo)`. Returns an expression, so nesting gives multiple integrals. |
| `Nabla.verify_integral(integrand, primitive, var)` | Differentiate the answer back. `:symbolic` is a certificate, `:numeric` is not. |
| `Nabla.limit(num, den, var, at:)` | L'Hopital. Indeterminacy is detected numerically — see Limits. |
| `Nabla.substitute(expr, var, replacement)` | Replace a variable, simplifying as it goes. |
| `Nabla.free_of?(expr, var)` | Whether an expression is independent of a variable. |

Targets are `Param` nodes (scalar kernel arguments) or `Load` nodes (`buf[i]`, buffer reads).

## Limits

- **f64 only**, inherited from moissanite's scalar types.
- **Integration covers a narrow set of forms** (see the symbolic layer). No integration by parts,
  partial fractions, or general substitution; no Risch.
- **Limits detect indeterminate forms numerically**, which can be fooled by very small non-zero
  values. No series expansion.
- **`simplify` is not a normal form**, so many correct integrals verify only numerically.
- **Sequential recurrences are not differentiable here** (see Reductions).
- `min` / `max` / `abs` have no derivative at the kink. nabla follows the `fmin`/`fmax` convention
  and routes the adjoint to one side; that is a convention, not a theorem.
- Forward and reverse differ by a few ulp from each other (see Correctness).
- High-order derivatives are limited by compile time, not run time.

## Development

```sh
bundle install
bundle exec rake            # tests + rubocop
bundle exec rake examples   # every example asserts its own invariants
```

## License

MIT
