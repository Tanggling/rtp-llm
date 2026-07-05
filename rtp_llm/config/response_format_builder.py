import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException


DEFAULT_THINK_END_TAG = "</think>\n\n"


@dataclass(frozen=True)
class ReasoningFormat:
    """Server/model resolved reasoning envelope format used for grammar wrapping."""

    tag_end: Union[str, List[str], Dict[str, Any]] = DEFAULT_THINK_END_TAG
    suffix: str = ""

    @staticmethod
    def decode_env_tag(tag: str) -> str:
        return tag.encode("utf-8").decode("unicode_escape")

    @classmethod
    def from_generate_env_config(cls, generate_env_config: Any) -> "ReasoningFormat":
        raw_tag = (
            getattr(generate_env_config, "think_end_tag", DEFAULT_THINK_END_TAG)
            or DEFAULT_THINK_END_TAG
        )
        tag = cls.decode_env_tag(str(raw_tag))
        raw_token_id = getattr(generate_env_config, "think_end_token_id", -1)
        token_id = -1 if raw_token_id is None else int(raw_token_id)
        if token_id != -1:
            return cls(tag_end={"type": "token", "token": int(token_id)})
        return cls(tag_end=tag)

    @classmethod
    def from_model_type(cls, model_type: Optional[str]) -> Optional["ReasoningFormat"]:
        if not model_type:
            return None
        normalized = model_type.lower().replace("-", "_")
        if normalized in {
            "qwen_3",
            "qwen_3_moe",
            "qwen_3_tool",
            "qwen_tool",
            "qwen3_next",
            "qwen35_moe",
            "qwen35_dense",
            "qwen35_moe_mtp",
            "qwen3_coder_moe",
        }:
            return cls(tag_end="</think>", suffix="\n\n")
        if normalized in {
            "deepseek_r1",
            "deepseek_v31",
            "deepseek_v3_1",
            "deepseek_v32",
            "deepseek_v3_2",
            "deepseek_v4",
            "kimi_k2",
            "kimi_linear",
            "kimi_k25",
        }:
            return cls(tag_end="</think>")
        return None

    @classmethod
    def from_model_type_and_env_config(
        cls, model_type: Optional[str], generate_env_config: Any
    ) -> "ReasoningFormat":
        env_format = cls.from_generate_env_config(generate_env_config)
        if env_format.token_end_id() is not None:
            return env_format
        env_stop_text = env_format.stop_text()
        if env_stop_text and env_stop_text != DEFAULT_THINK_END_TAG:
            return env_format
        return cls.from_model_type(model_type) or env_format

    def token_end_id(self) -> Optional[int]:
        if not isinstance(self.tag_end, dict):
            return None
        if self.tag_end.get("type") != "token":
            return None
        token = self.tag_end.get("token")
        if isinstance(token, bool) or not isinstance(token, int):
            return None
        return int(token)

    def stop_text(self) -> Optional[str]:
        if not isinstance(self.tag_end, str):
            return None
        return self.tag_end + self.suffix

    def prefix_format(self, max_thinking_tokens: int) -> Dict[str, Any]:
        think_tag = {
            "type": "tag",
            "begin": "",
            "content": {
                "type": "any_text",
                "max_tokens": max_thinking_tokens,
            },
            "end": self.tag_end,
        }
        if not self.suffix:
            return think_tag
        return {
            "type": "sequence",
            "elements": [
                think_tag,
                {"type": "const_string", "value": self.suffix},
            ],
        }


class ResponseFormatBuilder:
    """Normalize response_format and typed grammar fields in-place on GenerateConfig."""

    def __init__(
        self, config: Any, reasoning_format: Optional[ReasoningFormat] = None
    ):
        self.config = config
        self.reasoning_format = reasoning_format or ReasoningFormat()

    def apply(self) -> None:
        self._project_response_format_to_grammar_fields()
        self._normalize_grammar_fields()
        self._validate_grammar_constraints()

        if not self.config.in_think_mode:
            return

        if self._reasoning_envelope_final_is_json() is not None:
            return

        self._wrap_grammar_with_reasoning_envelope()

    @classmethod
    def grammar_terminate_without_stop_token(cls, config: Any) -> bool:
        if config.json_schema is not None:
            return True
        final_is_json = cls(config)._reasoning_envelope_final_is_json()
        return final_is_json is True

    def _project_response_format_to_grammar_fields(self) -> None:
        """Project response_format onto typed fields and clear it; rf wins over stale extra_configs grammar."""
        rf = self.config.response_format
        if rf is None:
            return

        self.config.json_schema = None
        self.config.regex = None
        self.config.ebnf = None
        self.config.structural_tag = None

        if rf.type == "json_schema":
            self.config.json_schema = json.dumps(
                rf.json_schema.schema, ensure_ascii=False, separators=(",", ":")
            )
        elif rf.type == "json_object":
            self.config.json_schema = json.dumps(
                {"type": "object"}, separators=(",", ":")
            )
        elif rf.type == "regex":
            self.config.regex = rf.pattern
        elif rf.type == "ebnf":
            self.config.ebnf = rf.grammar
        elif rf.type == "structural_tag":
            self.config.structural_tag = json.dumps(
                rf.structural_tag, ensure_ascii=False, separators=(",", ":")
            )
        # rf.type == "text" leaves all grammar fields cleared.

        self.config.response_format = None

    def _normalize_grammar_fields(self) -> None:
        """Normalize grammar fields before they cross backend/RPC boundaries."""
        if isinstance(self.config.json_schema, dict):
            self.config.json_schema = json.dumps(
                self.config.json_schema, ensure_ascii=False, separators=(",", ":")
            )
        if isinstance(self.config.structural_tag, dict):
            self.config.structural_tag = json.dumps(
                self.config.structural_tag, ensure_ascii=False, separators=(",", ":")
            )

    def _validate_grammar_constraints(self) -> None:
        grammar_fields = {
            "json_schema": self.config.json_schema,
            "regex": self.config.regex,
            "ebnf": self.config.ebnf,
            "structural_tag": self.config.structural_tag,
        }
        for name, value in grammar_fields.items():
            if isinstance(value, str) and not value.strip():
                raise FtRuntimeException(
                    ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                    f"{name} must not be empty",
                )

        count = (
            (self.config.json_schema is not None)
            + (self.config.regex is not None)
            + (self.config.ebnf is not None)
            + (self.config.structural_tag is not None)
        )
        if count > 1:
            raise FtRuntimeException(
                ExceptionType.UNSUPPORTED_OPERATION,
                "only one grammar constraint (json_schema / regex / ebnf / "
                "structural_tag) may be set per request",
            )
        # NormalOutputDispatcher skips per-token matcher advance under beam search, causing schema-illegal tokens.
        if count > 0 and (
            self.config.has_num_beams() or self.config.num_return_sequences > 1
        ):
            raise FtRuntimeException(
                ExceptionType.UNSUPPORTED_OPERATION,
                "grammar-constrained decoding (json_schema / regex / ebnf / "
                "structural_tag) is not supported with beam search "
                "(num_beams > 1 or num_return_sequences > 1)",
            )

    @staticmethod
    def _loads_json_field(name: str, value: Union[str, Dict[str, Any]]) -> Any:
        if isinstance(value, dict):
            return value
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError) as e:
            raise FtRuntimeException(
                ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                f"{name} must be valid JSON: {str(e)}",
            )

    @classmethod
    def _has_bounded_region(cls, node: Any) -> bool:
        if isinstance(node, dict):
            if node.get("type") in ("any_text", "any_tokens") and node.get(
                "max_tokens"
            ) is not None:
                return True
            return any(cls._has_bounded_region(value) for value in node.values())
        if isinstance(node, list):
            return any(cls._has_bounded_region(item) for item in node)
        return False

    def _final_grammar_format_node(self) -> Optional[Dict[str, Any]]:
        if self.config.json_schema is not None:
            schema: Any
            if self.config.json_schema == "$$ANY$$":
                schema = True
            else:
                schema = self._loads_json_field(
                    "json_schema", self.config.json_schema
                )
            if not isinstance(schema, (dict, bool)):
                raise FtRuntimeException(
                    ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                    "json_schema must be a JSON object or boolean",
                )
            return {"type": "json_schema", "json_schema": schema, "style": "json"}
        if self.config.regex is not None:
            return {"type": "regex", "pattern": self.config.regex}
        if self.config.ebnf is not None:
            return {"type": "grammar", "grammar": self.config.ebnf}
        if self.config.structural_tag is not None:
            structural_tag = self._loads_json_field(
                "structural_tag", self.config.structural_tag
            )
            if not isinstance(structural_tag, dict):
                raise FtRuntimeException(
                    ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                    "structural_tag must be a JSON object",
                )
            format_node = structural_tag.get("format")
            if structural_tag.get("type") != "structural_tag" or not isinstance(
                format_node, dict
            ):
                raise FtRuntimeException(
                    ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                    "structural_tag must have type='structural_tag' and a format object",
                )
            if self._has_bounded_region(format_node):
                raise FtRuntimeException(
                    ExceptionType.UNSUPPORTED_OPERATION,
                    "reasoning grammar cannot wrap a final structural_tag that "
                    "already contains any_text/any_tokens max_tokens",
                )
            return format_node
        return None

    def _reasoning_envelope_final_is_json(self) -> Optional[bool]:
        if self.config.structural_tag is None:
            return None
        structural_tag = self._loads_json_field(
            "structural_tag", self.config.structural_tag
        )
        if not isinstance(structural_tag, dict):
            return None
        if structural_tag.get("type") != "structural_tag":
            return None
        format_node = structural_tag.get("format")
        if not isinstance(format_node, dict) or format_node.get("type") != "sequence":
            return None
        elements = format_node.get("elements")
        if not isinstance(elements, list):
            return None
        final_format = self._extract_reasoning_envelope_final_format(elements)
        if final_format is None:
            return None
        return final_format.get("type") == "json_schema"

    @classmethod
    def _extract_reasoning_envelope_final_format(
        cls, elements: list[Any]
    ) -> Optional[Dict[str, Any]]:
        if len(elements) < 2 or not cls._looks_like_reasoning_think_tag(elements[0]):
            return None
        final_index = 1
        if (
            len(elements) >= 3
            and isinstance(elements[1], dict)
            and elements[1].get("type") == "const_string"
        ):
            final_index = 2
        if len(elements) != final_index + 1:
            return None
        final_format = elements[final_index]
        if not isinstance(final_format, dict):
            return None
        if cls._has_bounded_region(final_format):
            return None
        return final_format

    @staticmethod
    def _looks_like_reasoning_think_tag(node: Any) -> bool:
        if not isinstance(node, dict):
            return False
        if node.get("type") != "tag" or node.get("begin") != "":
            return False
        content = node.get("content")
        if not isinstance(content, dict):
            return False
        return (
            content.get("type") == "any_text"
            and content.get("max_tokens") is not None
            and "end" in node
        )

    def _wrap_grammar_with_reasoning_envelope(self) -> None:
        final_format = self._final_grammar_format_node()
        if final_format is None:
            return
        reasoning_prefix = self.reasoning_format.prefix_format(
            self.config.max_thinking_tokens
        )
        if reasoning_prefix.get("type") == "sequence":
            elements = list(reasoning_prefix["elements"]) + [final_format]
        else:
            elements = [reasoning_prefix, final_format]
        envelope = {
            "type": "structural_tag",
            "format": {
                "type": "sequence",
                "elements": elements,
            },
        }
        self.config.structural_tag = json.dumps(
            envelope, ensure_ascii=False, separators=(",", ":")
        )
        self.config.json_schema = None
        self.config.regex = None
        self.config.ebnf = None
