from __future__ import annotations

import asyncio
from dataclasses import dataclass

try:
    import pynvml
except ImportError:  # pragma: no cover
    pynvml = None


@dataclass(slots=True)
class GPUSnapshot:
    name: str
    memory_total_mb: int | None = None
    memory_used_mb: int | None = None
    memory_free_mb: int | None = None
    utilization_gpu_percent: int | None = None
    power_draw_watts: float | None = None
    power_limit_watts: float | None = None


@dataclass(slots=True)
class GPUPeakSnapshot:
    peak_power_watts: float | None = None
    peak_vram_used_mb: int | None = None
    power_limit_watts: float | None = None


def read_gpu_snapshots() -> list[GPUSnapshot]:
    if pynvml is None:
        return []

    rows: list[GPUSnapshot] = []
    try:
        pynvml.nvmlInit()
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            power_draw = _read_power_usage(handle)
            power_limit = _read_power_limit(handle)
            rows.append(
                GPUSnapshot(
                    name=_decode_name(pynvml.nvmlDeviceGetName(handle)),
                    memory_total_mb=int(memory.total / (1024 * 1024)),
                    memory_used_mb=int(memory.used / (1024 * 1024)),
                    memory_free_mb=int(memory.free / (1024 * 1024)),
                    utilization_gpu_percent=int(utilization.gpu),
                    power_draw_watts=power_draw,
                    power_limit_watts=power_limit,
                )
            )
    except Exception:
        rows = []
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    return rows


class GPUPeakSampler:
    def __init__(self, interval_seconds: float = 0.2) -> None:
        self._interval_seconds = interval_seconds
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._peak_power_watts: float | None = None
        self._peak_vram_used_mb: int | None = None
        self._power_limit_watts: float | None = None

    async def start(self) -> None:
        if pynvml is None:
            return
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> GPUPeakSnapshot:
        if self._task is None:
            return GPUPeakSnapshot()

        self._capture()
        self._stop_event.set()
        await self._task
        self._task = None
        return GPUPeakSnapshot(
            peak_power_watts=self._peak_power_watts,
            peak_vram_used_mb=self._peak_vram_used_mb,
            power_limit_watts=self._power_limit_watts,
        )

    async def _run(self) -> None:
        while not self._stop_event.is_set():
            self._capture()
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._interval_seconds)
            except TimeoutError:
                continue

    def _capture(self) -> None:
        snapshots = read_gpu_snapshots()
        if not snapshots:
            return

        total_power = 0.0
        total_power_limit = 0.0
        total_vram_used = 0
        saw_power = False
        saw_power_limit = False
        saw_vram = False

        for snapshot in snapshots:
            if snapshot.power_draw_watts is not None:
                total_power += snapshot.power_draw_watts
                saw_power = True
            if snapshot.power_limit_watts is not None:
                total_power_limit += snapshot.power_limit_watts
                saw_power_limit = True
            if snapshot.memory_used_mb is not None:
                total_vram_used += snapshot.memory_used_mb
                saw_vram = True

        if saw_power:
            self._peak_power_watts = max(self._peak_power_watts or 0.0, total_power)
        if saw_power_limit:
            self._power_limit_watts = total_power_limit
        if saw_vram:
            self._peak_vram_used_mb = max(self._peak_vram_used_mb or 0, total_vram_used)


def _decode_name(value: str | bytes) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _read_power_usage(handle: object) -> float | None:
    if pynvml is None:
        return None
    try:
        return round(float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000, 2)
    except Exception:
        return None


def _read_power_limit(handle: object) -> float | None:
    if pynvml is None:
        return None
    try:
        return round(float(pynvml.nvmlDeviceGetEnforcedPowerLimit(handle)) / 1000, 2)
    except Exception:
        try:
            return round(float(pynvml.nvmlDeviceGetPowerManagementLimit(handle)) / 1000, 2)
        except Exception:
            return None
