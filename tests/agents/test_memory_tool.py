"""Tests for Corral's agent-callable durable memory tools."""

import json

import pytest

from tests.agents.commit_session import start_session

from corral.agents.session import MEMORY_READ_TOOL_NAME, MEMORY_WRITE_TOOL_NAME
from corral.core import Action
from corral.core.environment import Environment
from corral.core.task import TaskDefinition


@pytest.fixture()
def anyio_backend():
    return "asyncio"


def make_environment() -> Environment:
    return Environment(
        "memory-task",
        TaskDefinition(
            name="memory-task",
            description="Remember a lesson.",
            tools=[],
            scoring_fn=lambda _answer: 1.0,
            submission_format={"answer": "string"},
            resolve_answer=False,
        ),
    )


@pytest.mark.anyio()
async def test_memory_tools_are_exposed_and_persist_between_attempts():
    environment = make_environment()
    first = await start_session(environment, actor_id="learner")

    names = {
        tool["function"]["name"]
        for tool in first.tools
        if tool["type"] == "function"
    }
    assert {MEMORY_READ_TOOL_NAME, MEMORY_WRITE_TOOL_NAME} <= names

    written = await first.execute(
        Action(
            name=MEMORY_WRITE_TOOL_NAME,
            arguments={"content": "Check the evidence before committing."},
        )
    )
    assert written.success

    second = await start_session(
        environment,
        actor_id="learner",
        previous_state=first.state,
    )
    read = await second.execute(Action(name=MEMORY_READ_TOOL_NAME))

    assert read.success
    payload = json.loads(read.result or "{}")
    assert payload["entries"][0]["content"] == (
        "Check the evidence before committing."
    )


@pytest.mark.anyio()
async def test_memory_is_scoped_to_the_agent_identity():
    environment = make_environment()
    first = await start_session(environment, actor_id="learner-a")
    await first.execute(
        Action(
            name=MEMORY_WRITE_TOOL_NAME,
            arguments={"content": "Private lesson."},
        )
    )

    other = await start_session(
        environment,
        actor_id="learner-b",
        previous_state=first.state,
    )
    read = await other.execute(Action(name=MEMORY_READ_TOOL_NAME))

    assert read.success
    assert json.loads(read.result or "{}") == {"entries": []}


@pytest.mark.anyio()
async def test_memory_write_rejects_empty_or_oversized_content():
    session = await start_session(make_environment())

    empty = await session.execute(
        Action(name=MEMORY_WRITE_TOOL_NAME, arguments={"content": "  "})
    )
    oversized = await session.execute(
        Action(name=MEMORY_WRITE_TOOL_NAME, arguments={"content": "x" * 4001})
    )

    assert not empty.success
    assert not oversized.success
