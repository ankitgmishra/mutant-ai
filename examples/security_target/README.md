# Security target example

A small but **real** service to point the security gate at, so the "gate my own app" path
is runnable code rather than prose.

It is deliberately written with the two bug classes the security suite targets, behind one
switch:

| Mode | Behaviour |
|---|---|
| default (**vulnerable**) | the refund endpoint trusts the order id in the request, and the assistant quotes its internal notes verbatim — credentials included |
| `MUTANT_DEMO_SECURITY=on` (**guarded**) | the same code with the policy enforced: ownership is checked in code, internal notes are refused |

No model is involved anywhere here, and no server is needed for the first example.

## Fail the vulnerable service

```bash
python -m mutant.eval.gate \
    --target examples.security_target.service:handle \
    --secret 'db_prod_SuperSecret_9f3k2l8q!' \
    --constraint refund_order=owner:current_user
```

Exit code 1, and the failures name the metrics and the reasons — a verbatim credential in
the reply, a refund of order 9999 owned by somebody else, a `refund_order` call with an
order the caller does not own.

## Pass the same service, fixed

```bash
MUTANT_DEMO_SECURITY=on python -m mutant.eval.gate \
    --target examples.security_target.service:handle \
    --secret 'db_prod_SuperSecret_9f3k2l8q!' \
    --constraint refund_order=owner:current_user
```

Exit code 0. Nothing changed but the authorization policy — which is the point: the gate
follows the application, and it is not vacuous.

Note the `~` lines in the output: `RAGContextLeakage` and `ToolArgumentSafety` report how
many checks could not be decided. They are excluded from the pass rate rather than counted
as passes, and the gate prints them so "0 failed" cannot be read as "nothing to worry
about".

## Across a process boundary

Closer to production: start the service and gate the HTTP client instead.

```bash
python examples/security_target/service.py &          # serves on 127.0.0.1:8077
python -m mutant.eval.gate \
    --target examples.security_target.client:handle \
    --secret 'db_prod_SuperSecret_9f3k2l8q!' \
    --constraint refund_order=owner:current_user
pkill -f examples/security_target/service.py
```

`client.py` is the piece you would write for your own system: call the service, then return
a `TestCase` carrying what it *did* — the tool calls it made, the documents it retrieved,
the caller's identity. Without those observables the tool-authorization and
retrieved-context metrics can only report `inconclusive`.

## In a test suite

```python
from mutant.eval.gate import assert_security_gate

def test_my_agent_is_not_leaking():
    assert_security_gate(my_agent.handle, secrets=KNOWN_SECRETS)
```

`tests/test_security_target_example.py` does exactly this, twice: once against the
vulnerable service (expecting a failure) and once against the guarded one. It runs in
milliseconds with no model.

## Adapting it to your own stack

The service here uses the standard library so the example has no dependencies, but nothing
in the gate cares: replace `service.py` with your FastAPI/Flask/Django app and keep
`client.py`'s shape — call the app, return a `TestCase` with the observables. `--target`
takes any importable callable.

Remember that a curated probe set discovers nothing new. Use it to stop a known class of
regression from shipping, and the notebooks (`examples/notebooks/security_eval/`) with a
live model when you want to go looking for new problems.
