"""Tests for input validation in src/runner.py run_briefing.

These tests exercise only the validation path — they return before any
graph invocation, so no LLM calls or Tavily requests. Fast and free.
"""

from src.runner import run_briefing
import pytest

@pytest.mark.asyncio
async def test_empty_string_returns_invalid_topic():
    result = await run_briefing("")
    assert result.status == "failure"
    assert result.reason == "invalid_topic"

@pytest.mark.asyncio
async def test_whitespace_only_returns_invalid_topic():
    result = await run_briefing("   \t\n  ")
    assert result.status == "failure"
    assert result.reason == "invalid_topic"

@pytest.mark.asyncio
async def test_too_short_returns_invalid_topic():
    result = await run_briefing("AI")
    assert result.status == "failure"
    assert result.reason == "invalid_topic"

@pytest.mark.asyncio
async def test_exactly_14_chars_returns_invalid_topic():
    """Boundary: MIN is 15, so 14 should fail."""
    result = await run_briefing("x" * 14)
    assert result.status == "failure"
    assert result.reason == "invalid_topic"

@pytest.mark.asyncio
async def test_too_long_returns_invalid_topic():
    """Boundary: MAX is 500, so 501 should fail."""
    result = await run_briefing("x" * 501)
    assert result.status == "failure"
    assert result.reason == "invalid_topic"

@pytest.mark.asyncio
async def test_non_string_returns_invalid_topic():
    result = await run_briefing(12345)  # type: ignore[arg-type]
    assert result.status == "failure"
    assert result.reason == "invalid_topic"

@pytest.mark.asyncio
async def test_failure_result_has_no_thread_id_for_pre_invocation_failure():
    """Input validation happens before graph invocation, so there's no
    thread_id yet. This is expected."""
    result = await run_briefing("AI")
    assert result.status == "failure"
    assert result.thread_id is None