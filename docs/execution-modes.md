# Execution Modes

Back to [README](../README.md) · See also [evaluation.md](evaluation.md) and [cli-runtime-setup.md](cli-runtime-setup.md)

## Mode summary

| Mode | Behavior | Typical use |
| --- | --- | --- |
| `mock` | simulates `openclaw *` commands in process | unit tests and narrow backend checks |
| `multi` | routes each command family to its local backend or skill | default benchmark evaluation |
| `martin` | semantic alias for `multi`; the model directly operates the CLI while the simulator/local skill stack responds | agent-in-sandbox evaluation and trajectory collection |
| `real` | runs `openclaw *` through the real CLI subprocess path while keeping calendar, email, weather, tasks, and file commands in the routed skill runtime | integration testing |
| `hybrid` | runs a live OpenClaw gateway and the same routed skill stack, with optional online providers enabled for supported app families | realistic end-to-end evaluation |

`multi` is the default benchmark mode because it preserves interactive state across command families without requiring every task to depend on an online runtime. `martin` is a semantic alias for `multi` and maps to the same implementation.

## What changes by mode

The environment always uses the same task semantics and evaluator contract. What changes is the execution backend.

- In `mock`, the environment uses a narrow in-process mock path.
- In `multi`, commands are routed through the local skill/runtime stack.
- In `martin`, commands use the same routed local skill/runtime stack as `multi`.
- In `real`, `openclaw *` is executed through the real CLI subprocess path.
- In `hybrid`, `openclaw` runs against a live gateway while the rest of the routed stack remains in place.

## Routing vs provider realization

Execution routing and provider realization are separate concerns.

In `real` and `hybrid`, command families such as `calendar`, `email`, `weather`, `tasks`, and `file` still execute through the normal CLI routing path and therefore appear in traces, effects, and evaluator-visible state. The distinction is backend realization: `openclaw` runs through the real CLI or gateway path, while other families run through local skill implementations that can optionally trigger online provider side effects when configured.
