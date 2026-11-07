"""Z.ai (Zhipu AI) chat models."""

from __future__ import annotations

from typing import Any, Literal, TypeAlias, cast

import openai
from langchain_core.language_models import (
    LangSmithParams,
    LanguageModelInput,
    ModelProfile,
    ModelProfileRegistry,
)
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.utils import from_env, secret_from_env
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_zai._version import __version__
from langchain_zai.data._profiles import _PROFILES

DEFAULT_API_BASE = "https://open.bigmodel.cn/api/paas/v4"

_DictOrPydanticClass: TypeAlias = dict[str, Any] | type[BaseModel]
_DictOrPydantic: TypeAlias = dict[str, Any] | BaseModel


_MODEL_PROFILES = cast("ModelProfileRegistry", _PROFILES)


def _get_default_model_profile(model_name: str) -> ModelProfile:
    default = _MODEL_PROFILES.get(model_name) or {}
    return default.copy()


class ChatZAI(BaseChatOpenAI):
    """Z.ai (Zhipu AI) chat model integration to access GLM models.

    Setup:
        Install `langchain-zai` and set environment variable `ZAI_API_KEY`.

        ```bash
        pip install -U langchain-zai
        export ZAI_API_KEY="your-api-key"
        ```

    Key init args — completion params:
        model:
            Name of Z.ai (Zhipu AI) model to use, e.g. `'glm-5.3'`.
        temperature:
            Sampling temperature. Zhipu's API restricts this to `[0.0, 1.0]`.
        max_tokens:
            Max number of tokens to generate.
        reasoning_effort:
            Reasoning effort for thinking models, one of `'max'`, `'xhigh'`,
            `'high'`, `'medium'`, `'low'`, `'minimal'`, or `'none'`.

    Key init args — client params:
        timeout:
            Timeout for requests.
        max_retries:
            Max number of retries.
        api_key:
            Z.ai (Zhipu AI) API key. If not passed in will be read from env var
            `ZAI_API_KEY`.
        api_base:
            Base URL of the Zhipu API. Defaults to Zhipu's open platform
            (`https://open.bigmodel.cn/api/paas/v4`). Automatically read from
            env var `ZAI_API_BASE` if not provided, e.g. to target the Z.ai
            international endpoint (`https://api.z.ai/api/paas/v4`).

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        ```python
        from langchain_zai import ChatZAI

        model = ChatZAI(
            model="glm-5.3",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            # api_key="...",
            # other params...
        )
        ```

    Invoke:
        ```python
        messages = [
            ("system", "You are a helpful translator. Translate the user sentence to French."),
            ("human", "I love programming."),
        ]
        model.invoke(messages)
        ```

    Stream:
        ```python
        for chunk in model.stream(messages):
            print(chunk.text, end="")
        ```
        ```python
        stream = model.stream(messages)
        full = next(stream)
        for chunk in stream:
            full += chunk
        full
        ```

    Async:
        ```python
        await model.ainvoke(messages)

        # stream:
        # async for chunk in (await model.astream(messages))

        # batch:
        # await model.abatch([messages])
        ```

    Tool calling:
        ```python
        from pydantic import BaseModel, Field


        class GetWeather(BaseModel):
            '''Get the current weather in a given location'''

            location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


        class GetPopulation(BaseModel):
            '''Get the current population in a given location'''

            location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


        model_with_tools = model.bind_tools([GetWeather, GetPopulation])
        ai_msg = model_with_tools.invoke("Which city is hotter today and which is bigger: LA or NY?")
        ai_msg.tool_calls
        ```

        See `ChatZAI.bind_tools()` method for more. Note that Zhipu's API only
        accepts the string `tool_choice="auto"`: the dict/object form accepted
        by `bind_tools` is coerced to `"auto"` at request time, and any other
        string value (e.g. `"required"`) is passed through unchanged and will
        be rejected by the API.

    Structured output:
        ```python
        from typing import Optional

        from pydantic import BaseModel, Field


        class Joke(BaseModel):
            '''Joke to tell user.'''

            setup: str = Field(description="The setup of the joke")
            punchline: str = Field(description="The punchline to the joke")
            rating: int | None = Field(description="How funny the joke is, from 1 to 10")


        structured_model = model.with_structured_output(Joke)
        structured_model.invoke("Tell me a joke about cats")
        ```

        See `ChatZAI.with_structured_output()` for more.

    Reasoning:
        GLM models emit chain-of-thought into `reasoning_content`, surfaced on
        `additional_kwargs["reasoning_content"]` (and as `reasoning` content
        blocks). Thinking is enabled by default; disable it with:

        ```python
        from langchain_zai import ChatZAI

        model = ChatZAI(
            model="glm-5.3",
            extra_body={"thinking": {"type": "disabled"}},
        )
        ```

        ```python
        # Tune reasoning effort (GLM-5.2 and later):
        model = ChatZAI(
            model="glm-5.3",
            reasoning_effort="high",
        )
        ```

        !!! note

            Do not use the inherited `reasoning` argument with this model: it
            routes requests to the OpenAI Responses API, which Zhipu does not
            implement. Pass Zhipu-specific request parameters such as
            `thinking` through `extra_body` instead.

    Token usage:
        ```python
        ai_msg = model.invoke(messages)
        ai_msg.usage_metadata
        ```
        ```python
        {"input_tokens": 28, "output_tokens": 5, "total_tokens": 33}
        ```

        Cached prompt tokens reported by Zhipu
        (`prompt_tokens_details.cached_tokens`) are mapped to
        `input_token_details.cache_read`. When streaming, usage arrives
        automatically in the final chunk.

    Response metadata:
        ```python
        ai_msg = model.invoke(messages)
        ai_msg.response_metadata
        ```
    """  # noqa: E501

    model_name: str = Field(alias="model")
    """The name of the model, e.g. `'glm-5.3'`."""
    api_key: SecretStr | None = Field(
        default_factory=secret_from_env("ZAI_API_KEY", default=None),
    )
    """Z.ai (Zhipu AI) API key."""
    api_base: str = Field(
        alias="base_url",
        default_factory=from_env("ZAI_API_BASE", default=DEFAULT_API_BASE),
    )
    """Z.ai (Zhipu AI) API base URL.

    Automatically read from env variable `ZAI_API_BASE` if not provided.
    """

    model_config = ConfigDict(populate_by_name=True)

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-zai"

    @property
    def lc_secrets(self) -> dict[str, str]:
        """A map of constructor argument names to secret ids."""
        return {"api_key": "ZAI_API_KEY"}

    def _get_ls_params(
        self,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LangSmithParams:
        ls_params = super()._get_ls_params(stop=stop, **kwargs)
        ls_params["ls_provider"] = "zai"
        return ls_params

    @model_validator(mode="after")
    def _set_zai_version(self) -> Self:
        """Set package version in metadata.

        Named uniquely to avoid shadowing `BaseChatOpenAI._set_openai_chat_version`;
        Pydantic replaces same-named validators rather than chaining them.
        """
        self._add_version("langchain-zai", __version__)
        return self

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate necessary environment vars and client params."""
        if self.api_base == DEFAULT_API_BASE and not (
            self.api_key and self.api_key.get_secret_value()
        ):
            msg = "If using default api base, ZAI_API_KEY must be set."
            raise ValueError(msg)
        client_params: dict = {
            k: v
            for k, v in {
                "api_key": self.api_key.get_secret_value() if self.api_key else None,
                "base_url": self.api_base,
                "timeout": self.request_timeout,
                "max_retries": self.max_retries,
                "default_headers": self.default_headers,
                "default_query": self.default_query,
            }.items()
            if v is not None
        }

        if not (self.client or None):
            sync_specific: dict = {"http_client": self.http_client}
            self.root_client = openai.OpenAI(**client_params, **sync_specific)
            self.client = self.root_client.chat.completions
        if not (self.async_client or None):
            async_specific: dict = {"http_client": self.http_async_client}
            self.root_async_client = openai.AsyncOpenAI(
                **client_params,
                **async_specific,
            )
            self.async_client = self.root_async_client.chat.completions
        return self

    def _resolve_model_profile(self) -> ModelProfile | None:
        return _get_default_model_profile(self.model_name) or None

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        # Zhipu's API only accepts the string tool_choice="auto" and rejects
        # the dict/object form ({"type": "function", "function": {"name": ...}})
        # that bind_tools produces when forcing a specific tool — which is
        # exactly what with_structured_output() does. Coerce the unsupported
        # dict form to "auto". In the with_structured_output() case only a
        # single tool is bound, so the model reliably calls it even without
        # forcing.
        if isinstance(payload.get("tool_choice"), dict):
            payload["tool_choice"] = "auto"
        return payload

    def _create_chat_result(
        self,
        response: dict | openai.BaseModel,
        generation_info: dict | None = None,
    ) -> ChatResult:
        rtn = super()._create_chat_result(response, generation_info)

        if not isinstance(response, openai.BaseModel):
            return rtn

        for generation in rtn.generations:
            if generation.message.response_metadata is None:
                generation.message.response_metadata = {}
            generation.message.response_metadata["model_provider"] = "zai"

        choices = getattr(response, "choices", None)
        if choices and hasattr(choices[0].message, "reasoning_content"):
            rtn.generations[0].message.additional_kwargs["reasoning_content"] = choices[
                0
            ].message.reasoning_content

        return rtn

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk,
            default_chunk_class,
            base_generation_info,
        )

        if (choices := chunk.get("choices")) and generation_chunk:
            top = choices[0]
            if isinstance(generation_chunk.message, AIMessageChunk):
                generation_chunk.message.response_metadata = {
                    **generation_chunk.message.response_metadata,
                    "model_provider": "zai",
                }
                if (
                    reasoning_content := top.get("delta", {}).get("reasoning_content")
                ) is not None:
                    generation_chunk.message.additional_kwargs["reasoning_content"] = (
                        reasoning_content
                    )

        return generation_chunk

    def with_structured_output(
        self,
        schema: _DictOrPydanticClass | None = None,
        *,
        method: Literal[
            "function_calling",
            "json_mode",
            "json_schema",
        ] = "function_calling",
        include_raw: bool = False,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema: The output schema. Can be passed in as:

                - An OpenAI function/tool schema,
                - A JSON Schema,
                - A `TypedDict` class,
                - Or a Pydantic class.

                If `schema` is a Pydantic class then the model output will be a
                Pydantic instance of that class, and the model-generated fields will be
                validated by the Pydantic class. Otherwise the model output will be a
                dict and will not be validated.

                See `langchain_core.utils.function_calling.convert_to_openai_tool` for
                more on how to properly specify types and descriptions of schema fields
                when specifying a Pydantic or `TypedDict` class.

            method: The method for steering model generation, one of:

                - `'function_calling'`:
                    Uses Zhipu's tool-calling features.
                - `'json_mode'`:
                    Uses Zhipu's JSON mode (`response_format={"type": "json_object"}`).

                `'json_schema'` is silently remapped to `'function_calling'`: Zhipu's
                `response_format` supports only `'text'` and `'json_object'`, so there
                is no native JSON-schema mode.

            include_raw:
                If `False` then only the parsed structured output is returned.

                If an error occurs during model output parsing it will be raised.

                If `True` then both the raw model response (a `BaseMessage`) and the
                parsed model response will be returned.

                If an error occurs during output parsing it will be caught and returned
                as well.

                The final output is always a `dict` with keys `'raw'`, `'parsed'`, and
                `'parsing_error'`.

            strict:
                Whether to enable strict schema adherence when generating the
                function call. Zhipu's API does not document a `strict` field on
                function definitions; leave it as `None` (default) unless you
                have verified your account accepts it.

            kwargs: Additional keyword args aren't supported.

        Returns:
            A `Runnable` that takes same inputs as a
                `langchain_core.language_models.chat.BaseChatModel`. If `include_raw` is
                `False` and `schema` is a Pydantic class, `Runnable` outputs an instance
                of `schema` (i.e., a Pydantic object). Otherwise, if `include_raw` is
                `False` then `Runnable` outputs a `dict`.

                If `include_raw` is `True`, then `Runnable` outputs a `dict` with keys:

                - `'raw'`: `BaseMessage`
                - `'parsed'`: `None` if there was a parsing error, otherwise the type
                  depends on the `schema` as described above.
                - `'parsing_error'`: `BaseException | None`
        """
        # Zhipu's response_format supports only 'text' and 'json_object' — there
        # is no json_schema mode, so remap to function calling.
        if method == "json_schema":
            method = "function_calling"

        return super().with_structured_output(
            schema,
            method=method,
            include_raw=include_raw,
            strict=strict,
            **kwargs,
        )
