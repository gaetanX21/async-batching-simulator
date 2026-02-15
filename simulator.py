import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from typing import ClassVar, Generic, Hashable, Sequence, TypeVar
from uuid import uuid4

from plotting import (
    plot_batcher_history,
    plot_inference_engine_history,
)

# generic types
T = TypeVar("T")  # any type
H = TypeVar("H", bound=Hashable)  # hashable type

# for type clarity
Seq = int
Duration = float
Timestamp = float
TIMESTAMPED_BATCH = tuple[Timestamp, Sequence[T]]
TIMESTAMPED_LANES = tuple[Timestamp, list[H | None]]


@dataclass(frozen=True)
class DataDistribution:
    """Distribution of data to be generated."""

    short: ClassVar[Seq] = 1  # arbitrary length unit, sets reference
    long: int = 2  # e.g. twice as long as short
    prob_short: float = 0.5  # probability of short vs. long sequence

    # need jitter for smoother plots
    sigma: float = 0.1  # standard deviation of generation duration
    base_dt: float = 0.1  # mock second/token (reduce for faster simulation)

    def sample(self) -> tuple[Duration, Seq]:
        """Sample a sequence (short or long) and its generation duration."""
        seq_len = self.short if random.random() < self.prob_short else self.long
        duration = seq_len * self.base_dt * (1 + abs(random.gauss(0, self.sigma)))
        return duration, seq_len


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for simulation."""

    dist: DataDistribution  # distribution of sequence lengths (binary for simplicity)
    max_concurrent_requests: int  # infra load
    seq_per_batch: int  # RL setup
    total_sequences: int  # for more x-axis points

    @property
    def long_term_intensity(self) -> float:
        """Long-term intensity of requests (requests per second)."""
        T_base = (
            self.dist.prob_short * self.dist.short
            + (1 - self.dist.prob_short) * self.dist.long
        ) * self.dist.base_dt
        base_intensity = 1 / T_base
        return self.max_concurrent_requests * base_intensity

    @property
    def num_short_per_batch(self) -> float:
        """Steady-state number of short sequences per batch."""
        return self.seq_per_batch * self.dist.prob_short

    @property
    def short_load(self) -> float:
        """Steady-state short load (number short sequences inside inference engine)."""
        # Little's law
        short_load = self.dist.short * self.dist.prob_short
        long_load = self.dist.long * (1 - self.dist.prob_short)
        ratio = short_load / (short_load + long_load)
        return ratio * self.max_concurrent_requests

    @property
    def transient_regime_severity(self) -> float:
        """Severity of transient regime (higher = worse)."""
        seq_len_asymmetry = self.dist.long / self.dist.short
        concurrency_aggravation = self.max_concurrent_requests / self.seq_per_batch
        return seq_len_asymmetry * concurrency_aggravation

    @property
    def num_batches(self) -> int:
        """Number of batches in simulation."""
        return self.total_sequences // self.seq_per_batch

    @property
    def simulation_duration(self) -> float:
        """Approximate duration of simulation (seconds)."""
        return self.total_sequences / self.long_term_intensity

    @property
    def batch_duration(self) -> float:
        """Duration of a batch in steady state (seconds)."""
        return self.seq_per_batch / self.long_term_intensity


@dataclass(frozen=True)
class GenerationRequest:
    seq: Seq
    id: str = field(default_factory=lambda: str(uuid4()))


class ThreadSafeBatcher(Generic[T]):
    def __init__(self, size: int):
        self._size = size
        self._lock = threading.Lock()
        self._buffer: list[T] = []
        self._buffer_history: list[TIMESTAMPED_BATCH] = []

    def add(self, item: T) -> None:
        with self._lock:
            self._buffer.append(item)
            if len(self._buffer) == self._size:
                self._flush()

    def _flush(self) -> None:
        self._buffer_history.append((time.time(), self._buffer))  # no need for copy
        self._buffer = []

    def flush_remaining(self) -> None:
        with self._lock:
            if self._buffer:
                self._flush()

    def get_history(self) -> Sequence[TIMESTAMPED_BATCH]:
        with self._lock:
            return deepcopy(self._buffer_history)


class ThreadSafeLanes(Generic[H]):
    def __init__(self, size: int):
        self._size = size
        self._lock = threading.Lock()
        self._lanes: list[H | None] = [None] * self._size
        self._lanes_history: list[TIMESTAMPED_LANES] = []
        self._update_lanes_history()
        self._free_lanes: set[int] = set(range(self._size))
        self._item_to_lane: dict[H, int] = {}  # track which lane holds which item

    def _update_lanes_history(self) -> None:
        self._lanes_history.append((time.time(), self._lanes.copy()))

    def add(self, item: H) -> None:
        with self._lock:
            if not self._free_lanes:
                raise RuntimeError(f"All {self._size} lanes full")
            lane_idx = min(self._free_lanes)  # pick lowest free lane
            self._lanes[lane_idx] = item
            self._item_to_lane[item] = lane_idx
            self._free_lanes.remove(lane_idx)
            self._update_lanes_history()

    def pop(self, item: H) -> None:
        with self._lock:
            if item not in self._item_to_lane:
                raise RuntimeError(f"Item {item} not in any lane")
            lane_idx = self._item_to_lane[item]
            self._lanes[lane_idx] = None
            del self._item_to_lane[item]
            self._free_lanes.add(lane_idx)
            self._update_lanes_history()

    def get_history(self) -> Sequence[TIMESTAMPED_LANES]:
        with self._lock:
            return deepcopy(self._lanes_history)


def generate_seq(
    dist: DataDistribution,
    batcher: ThreadSafeBatcher[Seq],
    inference_engine: ThreadSafeLanes[GenerationRequest],
) -> None:
    """Simulate generation of a sequence (short or long)."""
    gen_duration, seq = dist.sample()
    gen_request = GenerationRequest(seq=seq)
    inference_engine.add(gen_request)
    time.sleep(gen_duration)
    inference_engine.pop(gen_request)
    batcher.add(seq)


def simulate(
    config: SimulationConfig,
) -> tuple[Sequence[TIMESTAMPED_BATCH], Sequence[TIMESTAMPED_LANES]]:
    """Simulate continuous batching."""
    # init batcher
    batcher = ThreadSafeBatcher[Seq](size=config.seq_per_batch)

    # init inference engine
    max_num_seqs = config.max_concurrent_requests  # for clarity
    inference_engine = ThreadSafeLanes[GenerationRequest](size=max_num_seqs)  # mock

    # per-thread generation workload
    generate = partial(
        generate_seq,
        dist=config.dist,
        batcher=batcher,
        inference_engine=inference_engine,
    )

    # run simulation
    with ThreadPoolExecutor(max_workers=config.max_concurrent_requests) as executor:
        futures = [executor.submit(generate) for _ in range(config.total_sequences)]
        # Wait for all tasks to complete
        for future in futures:
            future.result()

    # flush any remaining sequences in batcher
    batcher.flush_remaining()

    # return results
    return batcher.get_history(), inference_engine.get_history()


def main() -> None:
    # simulation setup
    simulation_config = SimulationConfig(
        dist=DataDistribution(long=4, prob_short=0.75, sigma=0.1, base_dt=0.1),
        max_concurrent_requests=480,  # infra load
        seq_per_batch=144,  # RL setup
        total_sequences=10_000,  # for more x-axis points
    )

    # run simulation
    batcher_history, inference_engine_history = simulate(config=simulation_config)

    # plot results (saved to ./plots/)
    plot_batcher_history(batcher_history, simulation_config, output="save")
    plot_inference_engine_history(
        inference_engine_history, simulation_config, output="save"
    )


if __name__ == "__main__":
    main()
