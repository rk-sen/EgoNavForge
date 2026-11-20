"""Module type registry."""
from __future__ import annotations

from typing import Callable, Type

_REGISTRY: dict[str, Type] = {}


def register(type_name: str) -> Callable[[Type], Type]:
    def deco(cls: Type) -> Type:
        if type_name in _REGISTRY and _REGISTRY[type_name] is not cls:
            raise ValueError(f"duplicate module type: {type_name}")
        _REGISTRY[type_name] = cls
        cls.type_name = type_name  # type: ignore[attr-defined]
        return cls

    return deco


def create(type_name: str, cfg: dict):
    if type_name not in _REGISTRY:
        raise KeyError(
            f"unknown module type '{type_name}'. registered={sorted(_REGISTRY)}"
        )
    return _REGISTRY[type_name](cfg)


def registered() -> list[str]:
    return sorted(_REGISTRY)
