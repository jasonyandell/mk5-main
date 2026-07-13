"""Tiny logged-output algebra for burl.lab phase handlers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from .transcript import Move

Out = TypeVar("Out")
Next = TypeVar("Next")


@dataclass(frozen=True)
class Trace(Generic[Out]):
    """A pure result: journalable events plus an optional output value."""

    events: tuple[Move, ...] = ()
    output: Out | None = None

    @classmethod
    def empty(cls) -> "Trace[Out]":
        return cls()

    @classmethod
    def emit(cls, *events: Move) -> "Trace[Out]":
        return cls(events=tuple(events))

    @classmethod
    def pure(cls, output: Out) -> "Trace[Out]":
        return cls(output=output)

    def append(self, *events: Move) -> "Trace[Out]":
        return Trace(events=self.events + tuple(events), output=self.output)

    def map(self, fn: Callable[[Out], Next]) -> "Trace[Next]":
        if self.output is None:
            return Trace(events=self.events)
        return Trace(events=self.events, output=fn(self.output))

    def then(self, fn: Callable[[Out], "Trace[Next]"]) -> "Trace[Next]":
        if self.output is None:
            return Trace(events=self.events)
        tail = fn(self.output)
        return Trace(events=self.events + tail.events, output=tail.output)


__all__ = ["Trace"]
