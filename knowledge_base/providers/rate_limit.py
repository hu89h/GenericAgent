"""Small process-wide sliding-window limiters for KB model providers."""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable


@dataclass(eq=False, slots=True)
class Reservation:
    timestamp: float
    tokens: int
    requests: int = 1


class SlidingWindowRateLimiter:
    """Reserve minute and derived second capacity with configurable headroom."""

    def __init__(
        self,
        *,
        rpm: int,
        tpm: int,
        headroom: float = 0.8,
        window_seconds: float = 60.0,
        burst_window_seconds: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        headroom = min(0.95, max(0.1, float(headroom)))
        self.rpm = max(1, int(math.floor(max(1, int(rpm)) * headroom)))
        self.tpm = max(1, int(math.floor(max(1, int(tpm)) * headroom)))
        self.rps = max(1, int(math.floor(self.rpm / 60.0)))
        self.tps = max(1, int(math.floor(self.tpm / 60.0)))
        self.headroom = headroom
        self.window_seconds = max(0.1, float(window_seconds))
        self.burst_window_seconds = max(0.0, float(burst_window_seconds))
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._events: deque[Reservation] = deque()
        self._requests = 0
        self._tokens = 0

    def _purge(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._events and self._events[0].timestamp <= cutoff:
            event = self._events.popleft()
            self._requests -= event.requests
            self._tokens -= event.tokens

    @staticmethod
    def _totals(events) -> tuple[int, int]:
        return (
            sum(event.requests for event in events),
            sum(event.tokens for event in events),
        )

    @staticmethod
    def _constraint_wait(
        events,
        *,
        now: float,
        window: float,
        request_limit: int,
        token_limit: int,
        requested_tokens: int,
    ) -> list[float]:
        waits = []
        request_count, token_count = SlidingWindowRateLimiter._totals(events)
        if request_count + 1 > request_limit and events:
            waits.append(events[0].timestamp + window - now)
        if token_count + requested_tokens > token_limit:
            tokens_to_release = token_count + requested_tokens - token_limit
            released = 0
            for event in events:
                released += event.tokens
                if released >= tokens_to_release:
                    waits.append(event.timestamp + window - now)
                    break
        return waits

    def _wait_seconds(self, now: float, requested_tokens: int) -> float:
        waits = self._constraint_wait(
            list(self._events),
            now=now,
            window=self.window_seconds,
            request_limit=self.rpm,
            token_limit=self.tpm,
            requested_tokens=requested_tokens,
        )
        if self.burst_window_seconds:
            cutoff = now - self.burst_window_seconds
            recent = [event for event in self._events if event.timestamp > cutoff]
            waits.extend(
                self._constraint_wait(
                    recent,
                    now=now,
                    window=self.burst_window_seconds,
                    request_limit=self.rps,
                    token_limit=self.tps,
                    requested_tokens=requested_tokens,
                )
            )
        return max(0.01, max(waits, default=0.05))

    def acquire(self, estimated_tokens: int) -> Reservation:
        requested_tokens = max(1, int(estimated_tokens or 1))
        if requested_tokens > self.tpm:
            raise ValueError(
                f"request token estimate {requested_tokens} exceeds TPM budget {self.tpm}"
            )
        if self.burst_window_seconds and requested_tokens > self.tps:
            raise ValueError(
                f"request token estimate {requested_tokens} exceeds TPS budget {self.tps}"
            )
        while True:
            with self._lock:
                now = self._clock()
                self._purge(now)
                recent = (
                    [
                        event
                        for event in self._events
                        if event.timestamp > now - self.burst_window_seconds
                    ]
                    if self.burst_window_seconds
                    else []
                )
                recent_requests, recent_tokens = self._totals(recent)
                if (
                    self._requests + 1 <= self.rpm
                    and self._tokens + requested_tokens <= self.tpm
                    and (
                        not self.burst_window_seconds
                        or (
                            recent_requests + 1 <= self.rps
                            and recent_tokens + requested_tokens <= self.tps
                        )
                    )
                ):
                    reservation = Reservation(now, requested_tokens)
                    self._events.append(reservation)
                    self._requests += 1
                    self._tokens += requested_tokens
                    return reservation
                wait_seconds = self._wait_seconds(now, requested_tokens)
            # Wake periodically so configuration changes, clock jumps, and test
            # clocks do not leave a worker sleeping for a whole minute.
            self._sleep(min(wait_seconds, 1.0))

    def reconcile(self, reservation: Reservation, actual_tokens: int | None) -> None:
        if not actual_tokens:
            return
        actual = max(1, int(actual_tokens))
        with self._lock:
            now = self._clock()
            self._purge(now)
            if reservation in self._events:
                self._tokens += actual - reservation.tokens
                reservation.tokens = actual
                return
            # A request lasting longer than the window may have had its initial
            # reservation purged. Account for the completed response from now
            # without double-counting its RPM slot.
            completed = Reservation(now, actual, requests=0)
            self._events.append(completed)
            self._tokens += actual

    def snapshot(self) -> dict:
        with self._lock:
            self._purge(self._clock())
            return {
                "requests": self._requests,
                "tokens": self._tokens,
                "rpm_budget": self.rpm,
                "tpm_budget": self.tpm,
                "rps_budget": self.rps,
                "tps_budget": self.tps,
                "headroom": self.headroom,
            }


_limiters: dict[str, tuple[tuple[int, int, float], SlidingWindowRateLimiter]] = {}
_limiters_lock = threading.Lock()


def get_limiter(
    name: str,
    *,
    rpm: int,
    tpm: int,
    headroom: float = 0.8,
) -> SlidingWindowRateLimiter:
    """Return one shared limiter per provider quota bucket and configuration."""

    signature = (int(rpm), int(tpm), round(float(headroom), 6))
    with _limiters_lock:
        current = _limiters.get(str(name))
        if current is None or current[0] != signature:
            current = (
                signature,
                SlidingWindowRateLimiter(
                    rpm=signature[0],
                    tpm=signature[1],
                    headroom=signature[2],
                ),
            )
            _limiters[str(name)] = current
        return current[1]


__all__ = ["Reservation", "SlidingWindowRateLimiter", "get_limiter"]
