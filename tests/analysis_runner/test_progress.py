"""A run's progress reaches its provider, throttled, and only while the run goes on."""

from __future__ import annotations

import asyncio

import pytest

from model_service.analysis_runner.progress import reporting_progress


class Provider:
    """Takes progress sends, refusing them if told to."""

    def __init__(self, *, refuse: bool = False) -> None:
        self.sent: list[float] = []
        self.refuse = refuse

    async def send(self, fraction: float) -> None:
        if self.refuse:
            raise OSError("provider refused the progress")
        self.sent.append(fraction)

    async def until_sent(self, count: int) -> None:
        while len(self.sent) < count:
            await asyncio.sleep(0.001)


def test_the_latest_progress_is_sent_once_it_moved() -> None:
    provider = Provider()

    async def scenario() -> None:
        async with reporting_progress(provider.send, interval_seconds=0.01) as report:
            # Reported between two sends: only the latest goes out.
            for fraction in (0.1, 0.2, 0.3):
                report(fraction)
            await asyncio.wait_for(provider.until_sent(1), 2)
            # Less than a percentage point is not worth a send.
            report(0.305)
            await asyncio.sleep(0.05)
            report(0.6)
            await asyncio.wait_for(provider.until_sent(2), 2)

    asyncio.run(scenario())

    # A finished run is complete, whatever the last interval caught.
    assert provider.sent == [0.3, 0.6, 1.0]


def test_nothing_is_sent_after_the_run_ends() -> None:
    provider = Provider()

    async def scenario() -> None:
        async with reporting_progress(provider.send, interval_seconds=0.01) as report:
            report(0.5)
        report(0.9)
        await asyncio.sleep(0.05)

    asyncio.run(scenario())

    assert provider.sent == [1.0]


def test_a_failed_run_keeps_its_last_progress() -> None:
    provider = Provider()

    async def scenario() -> None:
        async with reporting_progress(provider.send, interval_seconds=0.01) as report:
            report(0.5)
            await asyncio.wait_for(provider.until_sent(1), 2)
            raise RuntimeError("the run failed")

    with pytest.raises(RuntimeError):
        asyncio.run(scenario())

    assert provider.sent == [0.5]


def test_a_refused_send_does_not_fail_the_run() -> None:
    provider = Provider(refuse=True)

    async def scenario() -> str:
        async with reporting_progress(provider.send, interval_seconds=0.01) as report:
            report(0.5)
            await asyncio.sleep(0.03)
        return "done"

    assert asyncio.run(scenario()) == "done"
