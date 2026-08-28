# Contributing

Issues and pull requests are welcome.

## Running the tests

```sh
pip install -e ".[speed,test]"
pytest
```

`tests/reference/expected.npz` holds the input grid and the expected outputs
for the example basin, so the tests run on a fresh clone with numpy alone.

## What the tests check

The structures and Strahler stream order must match the stored outputs cell
for cell. Drainage area and longest upstream path are allowed a floating-point
tolerance, because summation order at a confluence differs between methods.
The tests also check that every ordering, layering and safe manner agrees on
the same input, that only the conflict-free downstream layering has zero
receiver conflicts, and that a push under the other two layerings loses
writes.

## If you change a kernel or a structure

Run `python example.py` and check it still ends in PASS, then run `pytest`.
Do not regenerate `expected.npz` to make a failing test pass; a change that
alters the six structures is a versioned change and needs review first.

## Style

Numpy-style docstrings, one blank line between top-level definitions, and
comments that say why rather than what. New methods carry their origin and
complexity in the docstring and get an entry in `docs/methods.md`.
