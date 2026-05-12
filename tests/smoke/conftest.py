"""Smoke-test fixtures: reset rate-limiter bucket between tests."""
import pytest
import fraud_hunter_env.server.app as _app_module


@pytest.fixture(autouse=True)
def _reset_rate_limit_buckets():
    """Clear per-IP token buckets so each smoke test starts with a full burst."""
    for mw in getattr(_app_module.app, "user_middleware", []):
        instance = getattr(mw, "kwargs", {}).get("app") or getattr(mw, "app", None)
        if hasattr(instance, "_buckets"):
            instance._buckets.clear()

    # Also patch the burst to a very high value so rapid sequential
    # requests in a single test never hit the limiter.
    original_burst = _app_module._RATE_LIMIT_BURST
    original_rps = _app_module._RATE_LIMIT_RPS
    _app_module._RATE_LIMIT_BURST = 10_000.0
    _app_module._RATE_LIMIT_RPS = 10_000.0
    yield
    _app_module._RATE_LIMIT_BURST = original_burst
    _app_module._RATE_LIMIT_RPS = original_rps
