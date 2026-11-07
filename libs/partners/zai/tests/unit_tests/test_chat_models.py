"""Test chat model integration."""

from __future__ import annotations

from typing import Any, Literal
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.runnables import RunnableBinding
from langchain_tests.unit_tests import ChatModelUnitTests
from openai import BaseModel
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import PromptTokensDetails
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field, SecretStr

from langchain_zai.chat_models import ChatZAI

MODEL_NAME = "glm-5.3"

PROMPT_TOKENS = 100
COMPLETION_TOKENS = 10
TOTAL_TOKENS = 110
CACHED_TOKENS = 50


class MockOpenAIResponse(BaseModel):
    """Mock OpenAI response model."""

    choices: list
    error: None = None

    def model_dump(  # type: ignore[override]
        self,
        *,
        mode: Literal["json", "python"] | str = "python",  # noqa: PYI051
        include: Any = None,
        exclude: Any = None,
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        round_trip: bool = False,
        warnings: Literal["none", "warn", "error"] | bool = True,
        context: dict[str, Any] | None = None,
        serialize_as_any: bool = False,
    ) -> dict[str, Any]:
        """Convert to dictionary, ensuring `reasoning_content` is included."""
        choices_list = []
        for choice in self.choices:
            message_dict = {
                "role": "assistant",
                "content": choice.message.content,
            }
            # Add reasoning_content if present
            if hasattr(choice.message, "reasoning_content"):
                message_dict["reasoning_content"] = choice.message.reasoning_content
            choices_list.append({"message": message_dict})

        return {"choices": choices_list, "error": self.error}


class TestChatZAIUnit(ChatModelUnitTests):
    """Standard unit tests for `ChatZAI` chat model."""

    @property
    def chat_model_class(self) -> type[ChatZAI]:
        """Chat model class being tested."""
        return ChatZAI

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        """Parameters to initialize from environment variables."""
        return (
            {
                "ZAI_API_KEY": "api_key",
                "ZAI_API_BASE": "api_base",
            },
            {
                "model": MODEL_NAME,
            },
            {
                "api_key": "api_key",
                "api_base": "api_base",
            },
        )

    @property
    def chat_model_params(self) -> dict:
        """Parameters to create chat model instance for testing."""
        return {
            "model": MODEL_NAME,
            "api_key": "api_key",
        }

    def get_chat_model(self) -> ChatZAI:
        """Get a chat model instance for testing."""
        return ChatZAI(**self.chat_model_params)


class TestChatZAICustomUnit:
    """Custom tests specific to Z.ai (Zhipu AI) chat model."""

    def test_base_url_alias(self) -> None:
        """Test that `base_url` is accepted as an alias for `api_base`."""
        chat_model = ChatZAI(
            model=MODEL_NAME,
            api_key=SecretStr("api_key"),
            base_url="http://example.test/v1",
        )
        assert chat_model.api_base == "http://example.test/v1"

    def test_create_chat_result_with_reasoning_content(self) -> None:
        """Test that reasoning_content is properly extracted from response."""
        chat_model = ChatZAI(model=MODEL_NAME, api_key=SecretStr("api_key"))
        mock_message = MagicMock()
        mock_message.content = "Main content"
        mock_message.reasoning_content = "This is the reasoning content"
        mock_message.role = "assistant"
        mock_response = MockOpenAIResponse(
            choices=[MagicMock(message=mock_message)],
            error=None,
        )

        result = chat_model._create_chat_result(mock_response)
        assert (
            result.generations[0].message.additional_kwargs.get("reasoning_content")
            == "This is the reasoning content"
        )
        assert (
            result.generations[0].message.response_metadata.get("model_provider")
            == "zai"
        )

    def test_convert_chunk_with_reasoning_content(self) -> None:
        """Test that reasoning_content is properly extracted from streaming chunk."""
        chat_model = ChatZAI(model=MODEL_NAME, api_key=SecretStr("api_key"))
        chunk: dict[str, Any] = {
            "choices": [
                {
                    "delta": {
                        "content": "Main content",
                        "reasoning_content": "Streaming reasoning content",
                    },
                },
            ],
        }

        chunk_result = chat_model._convert_chunk_to_generation_chunk(
            chunk,
            AIMessageChunk,
            None,
        )
        if chunk_result is None:
            msg = "Expected chunk_result not to be None"
            raise AssertionError(msg)
        assert (
            chunk_result.message.additional_kwargs.get("reasoning_content")
            == "Streaming reasoning content"
        )

    def test_convert_chunk_without_reasoning(self) -> None:
        """Test that chunk without reasoning fields works correctly."""
        chat_model = ChatZAI(model=MODEL_NAME, api_key=SecretStr("api_key"))
        chunk: dict[str, Any] = {"choices": [{"delta": {"content": "Main content"}}]}

        chunk_result = chat_model._convert_chunk_to_generation_chunk(
            chunk,
            AIMessageChunk,
            None,
        )
        if chunk_result is None:
            msg = "Expected chunk_result not to be None"
            raise AssertionError(msg)
        assert chunk_result.message.additional_kwargs.get("reasoning_content") is None

    def test_convert_chunk_with_empty_delta(self) -> None:
        """Test that chunk with empty delta works correctly."""
        chat_model = ChatZAI(model=MODEL_NAME, api_key=SecretStr("api_key"))
        chunk: dict[str, Any] = {"choices": [{"delta": {}}]}

        chunk_result = chat_model._convert_chunk_to_generation_chunk(
            chunk,
            AIMessageChunk,
            None,
        )
        if chunk_result is None:
            msg = "Expected chunk_result not to be None"
            raise AssertionError(msg)
        assert chunk_result.message.additional_kwargs.get("reasoning_content") is None

    def test_payload_converts_dict_tool_choice_to_auto(self) -> None:
        """Test that dict-form tool_choice is converted to 'auto' in payload."""
        chat_model = ChatZAI(model=MODEL_NAME, api_key=SecretStr("api_key"))
        # Simulate the with_structured_output flow: bind_tools converts a tool
        # name string into the dict form {"type": "function", "function": ...}.
        bound = chat_model.bind_tools([SampleTool], tool_choice="SampleTool")
        messages = [("user", "test")]
        bound_kwargs = bound.kwargs  # type: ignore[attr-defined]

        # At bind_tools level, the parent converts the tool name to dict form
        assert isinstance(bound_kwargs.get("tool_choice"), dict)

        # But _get_request_payload should convert it to "auto"
        request_payload = chat_model._get_request_payload(messages, **bound_kwargs)
        assert request_payload.get("tool_choice") == "auto"

    def test_payload_preserves_string_tool_choice(self) -> None:
        """Test that string tool_choice values are passed through unchanged.

        Only the dict/object form is coerced to "auto" (Zhipu rejects it);
        string values are the user's explicit choice and are passed through
        as-is — values other than "auto" will be rejected by the API.
        """
        chat_model = ChatZAI(model=MODEL_NAME, api_key=SecretStr("api_key"))
        messages = [("user", "test")]

        for choice in ("auto", "required"):
            bound = chat_model.bind_tools([SampleTool], tool_choice=choice)
            request_payload = chat_model._get_request_payload(
                messages,
                **bound.kwargs,  # type: ignore[attr-defined]
            )
            assert request_payload.get("tool_choice") == choice, (
                f"Expected '{choice}' to be preserved, got "
                f"{request_payload.get('tool_choice')!r}"
            )

    def test_usage_metadata_cached_tokens_mapped_to_cache_read(self) -> None:
        """Test that Zhipu's nested cached_tokens surfaces as cache_read.

        Zhipu reports cached prompt tokens in the OpenAI-nested form
        (`prompt_tokens_details.cached_tokens`), which `BaseChatOpenAI` maps
        to `input_token_details.cache_read` — no Zhipu-specific handling is
        needed, and this test locks that in.
        """
        chat_model = ChatZAI(model=MODEL_NAME, api_key=SecretStr("api_key"))
        response = ChatCompletion(
            id="chatcmpl-test",
            created=0,
            model=MODEL_NAME,
            object="chat.completion",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="Hi"),
                ),
            ],
            usage=CompletionUsage(
                prompt_tokens=PROMPT_TOKENS,
                completion_tokens=COMPLETION_TOKENS,
                total_tokens=TOTAL_TOKENS,
                prompt_tokens_details=PromptTokensDetails(cached_tokens=CACHED_TOKENS),
            ),
        )

        result = chat_model._create_chat_result(response)

        message = result.generations[0].message
        assert isinstance(message, AIMessage)
        assert message.usage_metadata is not None
        assert message.usage_metadata["input_tokens"] == PROMPT_TOKENS
        assert (
            message.usage_metadata["input_token_details"]["cache_read"] == CACHED_TOKENS
        )


class SampleTool(PydanticBaseModel):
    """Sample tool schema for testing."""

    value: str = Field(description="A test value")


_MAX_RUNNABLE_DEPTH = 6


def _first_runnable_binding(runnable: Any, depth: int = 0) -> RunnableBinding | None:
    """Walk a composed runnable and return the first `RunnableBinding` found."""
    if isinstance(runnable, RunnableBinding):
        return runnable
    if depth > _MAX_RUNNABLE_DEPTH:
        return None
    for attr in ("bound", "first", "last", "steps"):
        value = getattr(runnable, attr, None)
        if value is None:
            continue
        candidates = value if isinstance(value, (list, tuple)) else [value]
        for candidate in candidates:
            found = _first_runnable_binding(candidate, depth + 1)
            if found is not None:
                return found
    return None


def test_with_structured_output_json_schema_remaps_to_function_calling() -> None:
    """Test that method='json_schema' is remapped to function calling.

    Zhipu's `response_format` supports only `'text'` and `'json_object'`, so
    the json_schema path must not bind a `response_format`; instead tools are
    bound, exactly as in the function_calling path.
    """
    llm = ChatZAI(model=MODEL_NAME, api_key=SecretStr("test_key"))
    structured = llm.with_structured_output(SampleTool, method="json_schema")
    assert structured is not None

    binding = _first_runnable_binding(structured)
    assert binding is not None
    assert binding.kwargs.get("tools")
    assert "response_format" not in binding.kwargs


def test_profile() -> None:
    """Test that model profile is loaded correctly."""
    model = ChatZAI(model="glm-4.6", api_key=SecretStr("test_key"))
    assert model.profile is not None
    assert model.profile["reasoning_output"]
    assert not model.profile["tool_choice"]


def test_metadata_versions() -> None:
    """Test that metadata reports the correct version info."""
    llm = ChatZAI(model=MODEL_NAME, api_key=SecretStr("test_key"))
    assert llm.metadata is not None
    versions = llm.metadata["lc_versions"]
    assert "langchain-core" in versions
    assert "langchain-zai" in versions
    assert "langchain-openai" in versions
