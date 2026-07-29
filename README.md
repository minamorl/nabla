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

Targets are `Param` nodes (scalar kernel arguments) or `Load` nodes (`buf[i]`, buffer reads).

## Limits

- **f64 only**, inherited from moissanite's scalar types.
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
