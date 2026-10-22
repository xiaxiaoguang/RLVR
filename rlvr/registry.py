"""Name -> factory registries, so components are chosen and parameterized from config.

    ATOMS = Registry("atom")

    @ATOMS.register()
    def rotate(k: int = 1): ...

    ATOMS.build("rotate")                  # rotate()
    ATOMS.build({"name": "rotate", "k": 2})  # rotate(k=2)

Registered objects are factories; `build(spec)` calls one with the spec's
parameters and returns the product. New entries can live in any module listed
under the config key `imports`, which is imported before configs are validated.
"""
from __future__ import annotations

import difflib
import inspect
from typing import Any, Callable, Iterator


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, Any] = {}

    def add(self, name: str, obj: Any, override: bool = False) -> Any:
        if name in self._items and not override:
            raise KeyError(f"{self.kind} '{name}' already registered (pass override=True to replace)")
        self._items[name] = obj
        return obj

    def register(self, name: str | None = None, *, override: bool = False) -> Callable:
        def deco(obj):
            self.add(name or obj.__name__, obj, override)
            return obj
        return deco

    def get(self, name: str) -> Any:
        if name not in self._items:
            close = difflib.get_close_matches(name, self._items, n=3)
            hint = f"; did you mean {close}?" if close else f"; available: {sorted(self._items)}"
            raise KeyError(f"unknown {self.kind} '{name}'{hint}")
        return self._items[name]

    def build(self, spec: str | dict, **extra) -> Any:
        """spec: 'name' or {'name': ..., **params}."""
        if isinstance(spec, str):
            name, params = spec, {}
        else:
            params = dict(spec)
            if "name" not in params:
                raise KeyError(f"{self.kind} spec needs a 'name' key: {spec}")
            name = params.pop("name")
        factory = self.get(name)
        params.update(extra)
        try:
            inspect.signature(factory).bind(**params)
        except TypeError as e:
            raise TypeError(f"{self.kind} '{name}' {inspect.signature(factory)}: {e}") from None
        return factory(**params)

    def names(self) -> list[str]:
        return list(self._items)

    def __contains__(self, name) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)
