import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from nanobot.bus.events import InboundMessage


def test_process_message_prepends_ingest_result_to_current_turn(tmp_path) -> None:
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation.max_tokens = 1024

    with patch("nanobot.agent.loop.SubagentManager") as MockSubMgr:
        MockSubMgr.return_value.cancel_by_session = AsyncMock(return_value=0)
        loop = AgentLoop(bus=bus, provider=provider, workspace=tmp_path)

    msg = InboundMessage(
        channel="cli",
        sender_id="user",
        chat_id="direct",
        content="Please remember this file",
        media=[str(tmp_path / "notes.txt")],
        metadata={"ingest_candidates": [str(tmp_path / "notes.txt")]},
    )
    (tmp_path / "notes.txt").write_text("retrieval augmented generation", encoding="utf-8")

    loop.memory_consolidator.maybe_consolidate_by_tokens = AsyncMock()
    loop.runner.run = AsyncMock(return_value=MagicMock(
        final_content="done",
        tools_used=[],
        messages=[],
        usage={},
        stop_reason="completed",
    ))

    asyncio.run(loop._process_message(msg))

    kwargs = loop.runner.run.await_args.args[0]
    current_user = kwargs.initial_messages[-1]["content"]
    if isinstance(current_user, list):
        current_text = current_user[0]["text"]
    else:
        current_text = current_user
    assert "Knowledge Ingestion Result" in current_text
    assert "notes.txt" in current_text


def test_media_download_failure_note_marks_file_not_ingested() -> None:
    from nanobot.agent.loop import AgentLoop

    msg = InboundMessage(
        channel="weixin",
        sender_id="user",
        chat_id="chat",
        content="[file: Happy-LLM-0727.pdf: download failed; not ingested]",
        metadata={
            "download_failures": [{
                "name": "Happy-LLM-0727.pdf",
                "reason": "WeChat media download failed before knowledge ingestion",
            }],
        },
    )

    note = AgentLoop._media_download_failure_note(msg)

    assert note is not None
    assert "Happy-LLM-0727.pdf" in note
    assert "was not saved or parsed" in note
