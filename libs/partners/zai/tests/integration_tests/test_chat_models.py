"""Test ChatZAI chat model."""

from __future__ import annotations

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, BaseMessageChunk
from langchain_core.tools import BaseTool
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_zai.chat_models import ChatZAI

MODEL_NAME = "glm-5.3"


class TestChatZAI(ChatModelIntegrationTests):
    """Test `ChatZAI` chat model."""

    @property
    def chat_model_class(self) -> type[ChatZAI]:
        """Return class of chat model being tested."""
        return ChatZAI

    @property
    def chat_model_params(self) -> dict:
        """Parameters to create chat model instance for testing."""
        return {
            "model": MODEL_NAME,
            "temperature": 0,
        }

    @property
    def supports_json_mode(self) -> bool:
        """(bool) whether the chat model supports JSON mode."""
        return True

    @property
    def has_tool_choice(self) -> bool:
        """(bool) whether the model supports forcing tool choice.

        Zhipu's API accepts only `tool_choice="auto"`, so tool choice cannot
        be forced. Returning `False` makes the standard tests bind tools
        without forcing.
        """
        return False

    @pytest.mark.xfail(reason="Not yet supported.")
    def test_tool_message_histories_list_content(
        self,
        model: BaseChatModel,
        my_adder_tool: BaseTool,
    ) -> None:
        """Override test for tool message histories with list content."""
        super().test_tool_message_histories_list_content(model, my_adder_tool)


@pytest.mark.xfail(reason="Takes > 30s to run.")
def test_reasoning_content() -> None:
    """Test reasoning content."""
    # Thinking is enabled by default on GLM models.
    chat_model = ChatZAI(model="glm-5.3")
    response = chat_model.invoke("What is 3^3?")
    assert response.content
    assert response.additional_kwargs["reasoning_content"]

    content_blocks = response.content_blocks
    assert content_blocks is not None
    assert len(content_blocks) > 0
    reasoning_blocks = [
        block for block in content_blocks if block.get("type") == "reasoning"
    ]
    assert len(reasoning_blocks) > 0
    raise ValueError


@pytest.mark.xfail(reason="Takes > 30s to run.")
def test_reasoning_content_streaming() -> None:
    """Test reasoning content with streaming."""
    # Thinking is enabled by default on GLM models.
    chat_model = ChatZAI(model="glm-5.3")
    full: BaseMessageChunk | None = None
    for chunk in chat_model.stream("What is 3^3?"):
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert full.additional_kwargs["reasoning_content"]

    content_blocks = full.content_blocks
    assert content_blocks is not None
    assert len(content_blocks) > 0
    reasoning_blocks = [
        block for block in content_blocks if block.get("type") == "reasoning"
    ]
    assert len(reasoning_blocks) > 0
