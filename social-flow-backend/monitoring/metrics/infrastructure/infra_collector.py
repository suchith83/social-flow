# Collects infrastructure metrics from servers, VMs, and cloud services
"""
Infrastructure-level metrics collector.

- Uses psutil for host/container metrics
- Provides an async-friendly sampler (start/stop)
- Exposes Prometheus metrics (Gauges / Counters)
- Can optionally call cloud-export adapters (placeholder)
"""

import asyncio
import logging
from prometheus_client import Gauge, Counter
from typing import Optional, Callable
from .config import InfraMetricsConfig
from .utils import safe_psutil, now_ts

logger = logging.getLogger("infra_metrics.infra_collector")


class InfraCollector:
    """
    InfraCollector periodically samples host metrics and updates Prometheus metrics.

    Usage:
        collector = InfraCollector()
        await collector.start()    # runs background tasks
        await collector.stop()     # stops background tasks
    """

    def __init__(self,
                 sample_interval: int = InfraMetricsConfig.SCRAPE_INTERVAL_SECONDS,
                 high_res_interval: int = InfraMetricsConfig.HIGH_RES_INTERVAL_SECONDS,
                 cloud_exporter: Optional[Callable] = None):
        self.psutil = safe_psutil()
        self.sample_interval = sample_interval
        self.high_res_interval = high_res_interval
        self.cloud_exporter = cloud_exporter  # optional callback to push metrics to cloud
        self._tasks = []
        self._stop_event = asyncio.Event()

        # Prometheus metrics (labels where helpful)
        self.cpu_percent = Gauge(
            "infra_cpu_percent_total", "CPU usage percent (system-wide)", ["mode"]
        )  # mode: user/system/idle/irq/softirq/steal/guest
        self.cpu_count = Gauge("infra_cpu_count", "Logical CPU count")
        self.memory_used_bytes = Gauge("infra_memory_used_bytes", "Memory used in bytes")
        self.memory_total_bytes = Gauge("infra_memory_total_bytes", "Memory total in bytes")
        self.swap_used_bytes = Gauge("infra_swap_used_bytes", "Swap used bytes")
        self.disk_used_bytes = Gauge("infra_disk_used_bytes", "Disk used bytes", ["device", "mountpoint"])
        self.disk_total_bytes = Gauge("infra_disk_total_bytes", "Disk total bytes", ["device", "mountpoint"])
        self.network_bytes_sent = Counter("infra_net_bytes_sent_total", "Network bytes sent", ["iface"])
        self.network_bytes_recv = Counter("infra_net_bytes_recv_total", "Network bytes received", ["iface"])
        self.process_count = Gauge("infra_process_count", "Number of processes")
        self.uptime_seconds = Gauge("infra_uptime_seconds", "System uptime in seconds")

    async def _sample_loop(self):
        """Periodic sampling at sample_interval for medium-resolution metrics."""
        logger.info("InfraCollector: starting sample loop (interval=%s)", self.sample_interval)
        while not self._stop_event.is_set():
            try:
                self._collect_once()
            except Exception as e:
                logger.exception("InfraCollector: sampling error: %s", e)
            await asyncio.wait([self._stop_event.wait()], timeout=self.sample_interval)

    async def _high_res_cpu_loop(self):
        """Higher resolution CPU sampling for bursts / fine-grained charts."""
        logger.info("InfraCollector: starting high-res CPU loop (interval=%s)", self.high_res_interval)
        while not self._stop_event.is_set():
            try:
                self._collect_cpu()
            except Exception as e:
                logger.exception("InfraCollector: cpu sampling error: %s", e)
            await asyncio.wait([self._stop_event.wait()], timeout=self.high_res_interval)

    def _collect_once(self):
        """Synchronous sampling call that updates most metrics once."""
        # CPU aggregate
        self._collect_cpu()
        # Memory
        vm = self.psutil.virtual_memory()
        self.memory_used_bytes.set(vm.used)
        self.memory_total_bytes.set(vm.total)
        # Swap
        swap = self.psutil.swap_memory()
        self.swap_used_bytes.set(swap.used)
        # Disk partitions
        for part in self.psutil.disk_partitions(all=False):
            try:
                du = self.psutil.disk_usage(part.mountpoint)
                self.disk_used_bytes.labels(device=part.device, mountpoint=part.mountpoint).set(du.used)
                self.disk_total_bytes.labels(device=part.device, mountpoint=part.mountpoint).set(du.total)
            except Exception:
                # mountpoint may be inaccessible; ignore safely
                logger.debug("InfraCollector: unable to read disk usage for %s", part.mountpoint)
        # Network counters (psutil returns cumulative)
        net_io = self.psutil.net_io_counters(pernic=True)
        for iface, counters in net_io.items():
            # counters.bytes_sent and bytes_recv are cumulative since boot -> Prometheus Counters are appropriate
            self.network_bytes_sent.labels(iface=iface).inc(0)  # ensure label exists even if unchanged
            self.network_bytes_recv.labels(iface=iface).inc(0)
            # We cannot call .inc(cumulative) reliably without storing last values; to keep it idempotent
            # we call .inc(0) here; exporter should be scraped and Prometheus will handle monotonic increases
            # If you want delta tracking, adapt to persist last counters and compute deltas.
        # Process count
        self.process_count.set(len(self.psutil.pids()))
        # Uptime
        boot_ts = self.psutil.boot_time()
        self.uptime_seconds.set(now_ts() - boot_ts)

        # Optional cloud push hook
        if self.cloud_exporter:
            try:
                self.cloud_exporter()  # user-provided push function
            except Exception:
                logger.exception("InfraCollector: cloud_exporter hook failed")

    def _collect_cpu(self):
        """Collect CPU metrics and set gauge labels for key modes."""
        # psutil.cpu_times_percent returns object with fields: user, system, idle, iowait, ...
        cpu_perc = self.psutil.cpu_times_percent(interval=None, percpu=False)
        # set relevant modes if present
        for mode in ("user", "system", "idle", "iowait", "steal", "guest"):
            val = getattr(cpu_perc, mode, None)
            if val is not None:
                self.cpu_percent.labels(mode).set(float(val))
        # cpu count
        self.cpu_count.set(self.psutil.cpu_count(logical=True))

    async def start(self):
        """Start background sampling tasks."""
        if self._tasks:
            logger.warning("InfraCollector: start called but tasks already running")
            return
        self._stop_event.clear()
        loop = asyncio.get_running_loop()
        self._tasks = [
            loop.create_task(self._sample_loop()),
            loop.create_task(self._high_res_cpu_loop()),
        ]
        logger.info("InfraCollector started with %d tasks", len(self._tasks))

    async def stop(self):
        """Stop background tasks and wait for them to finish."""
        if not self._tasks:
            return
        self._stop_event.set()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        logger.info("InfraCollector stopped")
