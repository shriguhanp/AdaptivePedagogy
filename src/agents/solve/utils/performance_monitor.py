#!/usr/bin/env python
"""
Performance Monitor - Performance monitoring and tracking system
Records Agent execution time, Token consumption and other metrics
"""

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import wraps
import json
from pathlib import Path
import time
from typing import Any


@dataclass
class PerformanceMetrics:
    """Performance metrics"""

    # Basic information
    agent_name: str  # Agent name
    start_time: float  # Start time (timestamp)
    end_time: float | None = None  # End time (timestamp)
    duration: float | None = None  # Duration (seconds)

    # Token statistics
    prompt_tokens: int = 0  # Prompt token count
    completion_tokens: int = 0  # Generated token count
    total_tokens: int = 0  # Total token count

    # Request statistics
    api_calls: int = 0  # API call count
    errors: int = 0  # Error count

    # Custom metrics
    custom_metrics: dict[str, Any] = field(default_factory=dict)

    def mark_end(self):
        """Mark end time and calculate duration"""
        self.end_time = time.time()
        if self.start_time:
            self.duration = self.end_time - self.start_time

    def add_tokens(self, prompt: int = 0, completion: int = 0):
        """Add Token statistics"""
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens = self.prompt_tokens + self.completion_tokens

    def add_api_call(self):
        """Increment API call count"""
        self.api_calls += 1

    def add_error(self):
        """Increment error count"""
        self.errors += 1

    def set_custom_metric(self, key: str, value: Any):
        """Set custom metric"""
        self.custom_metrics[key] = value

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def summary(self) -> str:
        """Generate summary"""
        lines = [
            f"Agent: {self.agent_name}",
            f"Duration: {self.duration:.2f}s" if self.duration else "Duration: N/A",
            f"Total Tokens: {self.total_tokens}",
            f"  Prompt: {self.prompt_tokens}",
            f"  Completion: {self.completion_tokens}",
            f"API Calls: {self.api_calls}",
            f"Errors: {self.errors}",
        ]

        if self.custom_metrics:
            lines.append("Custom Metrics:")
            for k, v in self.custom_metrics.items():
                lines.append(f"  {k}: {v}")

        return "\n".join(lines)


class PerformanceMonitor:
    """Performance monitor"""

    def __init__(self, enabled: bool = True, save_dir: str | None = None):
        """
        Initialize performance monitor

        Args:
            enabled: Whether to enable monitoring
            save_dir: Save directory (None uses ./data/user/performance)
        """
        self.enabled = enabled

        if save_dir is None:
            # Default to data/user/performance (relative to project root)
            project_root = Path(__file__).parent.parent.parent.parent.parent
            self.save_dir = project_root / "data" / "user" / "performance"
        else:
            self.save_dir = Path(save_dir)

        if self.enabled:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # Current session metrics
        self.metrics: dict[str, PerformanceMetrics] = {}

        # Overall statistics
        self.total_duration = 0.0
        self.total_tokens = 0
        self.total_api_calls = 0
        self.total_errors = 0

    def start_tracking(self, agent_name: str) -> PerformanceMetrics:
        """
        Start tracking Agent performance

        Args:
            agent_name: Agent name

        Returns:
            PerformanceMetrics instance
        """
        if not self.enabled:
            return PerformanceMetrics(agent_name=agent_name, start_time=time.time())

        metrics = PerformanceMetrics(agent_name=agent_name, start_time=time.time())

        self.metrics[agent_name] = metrics
        return metrics

    def end_tracking(self, agent_name: str):
        """
        End tracking and update statistics

        Args:
            agent_name: Agent name
        """
        if not self.enabled:
            return

        if agent_name in self.metrics:
            metrics = self.metrics[agent_name]
            metrics.mark_end()

            # Update overall statistics
            if metrics.duration:
                self.total_duration += metrics.duration
                self.total_tokens += metrics.total_tokens
                self.total_api_calls += metrics.api_calls
                self.total_errors += metrics.errors

    @contextmanager
    def track(self, agent_name: str):
        """
        Context manager: automatically track Agent performance

        Args:
            agent_name: Agent name

        Usage:
            with monitor.track("agent_name"):
                # Agent execution logic
                ...
        """
        # Start tracking
        metrics = self.start_tracking(agent_name)
        try:
            yield metrics
        except Exception:
            # Record error
            if self.enabled:
                metrics.add_error()
            raise
        finally:
            # End tracking
            self.end_tracking(agent_name)

    def get_metrics(self, agent_name: str) -> PerformanceMetrics | None:
        """
        Get Agent performance metrics

        Args:
            agent_name: Agent name

        Returns:
            PerformanceMetrics instance, or None if not found
        """
        return self.metrics.get(agent_name)

    def get_all_metrics(self) -> dict[str, PerformanceMetrics]:
        """Get all metrics"""
        return self.metrics

    def get_summary(self) -> dict[str, Any]:
        """
        Get overall statistics summary

        Returns:
            {
                'total_duration': float,
                'total_tokens': int,
                'total_api_calls': int,
                'total_errors': int,
                'agents': Dict[str, Dict]
            }
        """
        return {
            "total_duration": self.total_duration,
            "total_tokens": self.total_tokens,
            "total_api_calls": self.total_api_calls,
            "total_errors": self.total_errors,
            "agents": {name: metrics.to_dict() for name, metrics in self.metrics.items()},
        }

    def generate_report(self) -> dict[str, Any]:
        """
        Generate performance report (alias for get_summary)

        Returns:
            Performance report dictionary
        """
        return self.get_summary()

    def save(self, filepath: str | None = None) -> str:
        """
        Save performance metrics to file

        Args:
            filepath: Save path (None for auto-generated)

        Returns:
            Saved file path
        """
        if not self.enabled:
            return ""

        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.save_dir / f"performance_{timestamp}.json"
        else:
            filepath = Path(filepath)

        summary = self.get_summary()

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        return str(filepath)

    def print_summary(self):
        """Print overall statistics summary"""
        print("=" * 60)
        print("Performance Monitoring Summary")
        print("=" * 60)
        print(f"Total Duration: {self.total_duration:.2f}s")
        print(f"Total Tokens: {self.total_tokens}")
        print(f"Total API Calls: {self.total_api_calls}")
        print(f"Total Errors: {self.total_errors}")
        print()
        print("Agent Details:")
        print("-" * 60)

        for agent_name, metrics in self.metrics.items():
            print(f"\n{agent_name}:")
            print(f"  Duration: {metrics.duration:.2f}s" if metrics.duration else "  Duration: N/A")
            print(f"  Tokens: {metrics.total_tokens}")
            print(f"  API Calls: {metrics.api_calls}")
            print(f"  Errors: {metrics.errors}")

        print("=" * 60)

    def reset(self):
        """Reset all statistics"""
        self.metrics.clear()
        self.total_duration = 0.0
        self.total_tokens = 0
        self.total_api_calls = 0
        self.total_errors = 0


# Decorator: automatically track Agent performance


def track_performance(monitor: PerformanceMonitor):
    """
    Performance tracking decorator

    Args:
        monitor: PerformanceMonitor instance

    Usage:
        @track_performance(monitor)
        async def my_agent_process(...):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            agent_name = getattr(self, "agent_name", func.__name__)

            # Start tracking
            metrics = monitor.start_tracking(agent_name)

            try:
                result = await func(self, *args, **kwargs)
                return result
            except Exception:
                metrics.add_error()
                raise
            finally:
                # End tracking
                monitor.end_tracking(agent_name)

        return wrapper

    return decorator


# Global monitor instance

_global_monitor: PerformanceMonitor | None = None


def get_monitor(enabled: bool = True, save_dir: str | None = None) -> PerformanceMonitor:
    """
    Get global monitor instance (singleton pattern)

    Args:
        enabled: Whether to enable
        save_dir: Save directory

    Returns:
        PerformanceMonitor instance
    """
    global _global_monitor

    if _global_monitor is None:
        _global_monitor = PerformanceMonitor(enabled=enabled, save_dir=save_dir)

    return _global_monitor


def init_monitor_from_config(config: dict) -> PerformanceMonitor:
    """
    Initialize monitor from configuration

    Args:
        config: Configuration dictionary

    Returns:
        PerformanceMonitor instance
    """
    monitoring_config = config.get("monitoring", {})

    enabled = monitoring_config.get("enabled", True)
    save_dir = monitoring_config.get("save_dir", "./logs/performance")

    return get_monitor(enabled=enabled, save_dir=save_dir)


if __name__ == "__main__":
    # Test performance monitoring
    print("Performance Monitoring Test")
    print("=" * 60)

    # Create monitor
    monitor = PerformanceMonitor(enabled=True)

    # Simulate Agent execution
    import asyncio

    async def simulate_agent(agent_name: str, duration: float, tokens: int):
        """Simulate Agent execution"""
        metrics = monitor.start_tracking(agent_name)

        # Simulate work
        await asyncio.sleep(duration)

        # Record tokens
        metrics.add_tokens(prompt=tokens // 2, completion=tokens // 2)
        metrics.add_api_call()

        monitor.end_tracking(agent_name)

    # Run simulation
    async def run_simulation():
        await simulate_agent("decompose_agent", 0.5, 100)
        await simulate_agent("rag_agent", 1.0, 500)
        await simulate_agent("plan_agent", 0.8, 300)
        await simulate_agent("execute_agent", 1.5, 800)

    asyncio.run(run_simulation())

    # Print summary
    monitor.print_summary()

    # Save results
    saved_path = monitor.save()
    print(f"\nPerformance metrics saved to: {saved_path}")
