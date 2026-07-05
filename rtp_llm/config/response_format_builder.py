from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.grammar_constraint import (
    GRAMMAR_FIELD_NAMES,
    GrammarConstraint,
    dump_compact_json,
    has_bounded_region,
    load_json_field,
    normalize_grammar_value,
)

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

    def __init__(self, config: Any, reasoning_format: Optional[ReasoningFormat] = None):
        self.config = config
        self.reasoning_format = reasoning_format or ReasoningFormat()

    def apply(self) -> None:
        constraint = self._resolve_grammar_constraint()

        if not self.config.in_think_mode:
            return

        if self._reasoning_envelope_final_is_json() is not None:
            return

        if constraint is not None:
            self._wrap_grammar_with_reasoning_envelope(constraint)

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

        self._clear_grammar_fields()
        constraint = GrammarConstraint.from_response_format(rf)
        if constraint is not None:
            self._apply_constraint(constraint)

        self.config.response_format = None

    def _resolve_grammar_constraint(self) -> Optional[GrammarConstraint]:
        self._project_response_format_to_grammar_fields()
        self._normalize_grammar_fields()
        constraints = GrammarConstraint.collect_from_config(self.config)
        self._validate_grammar_constraints(constraints)
        if not constraints:
            return None
        return constraints[0]

    def _clear_grammar_fields(self) -> None:
        for name in GRAMMAR_FIELD_NAMES:
            setattr(self.config, name, None)

    def _apply_constraint(self, constraint: GrammarConstraint) -> None:
        for name, value in constraint.as_config_update().items():
            setattr(self.config, name, value)

    def _normalize_grammar_fields(self) -> None:
        """Normalize grammar fields before they cross backend/RPC boundaries."""
        for name in GRAMMAR_FIELD_NAMES:
            value = getattr(self.config, name)
            if value is not None:
                setattr(self.config, name, normalize_grammar_value(name, value))

    def _validate_grammar_constraints(
        self, constraints: List[GrammarConstraint]
    ) -> None:
        for constraint in constraints:
            constraint.validate_not_empty()

        if len(constraints) > 1:
            raise FtRuntimeException(
                ExceptionType.UNSUPPORTED_OPERATION,
                "only one grammar constraint (json_schema / regex / ebnf / "
                "structural_tag) may be set per request",
            )
        # NormalOutputDispatcher skips per-token matcher advance under beam search, causing schema-illegal tokens.
        if constraints and (
            self.config.has_num_beams() or self.config.num_return_sequences > 1
        ):
            raise FtRuntimeException(
                ExceptionType.UNSUPPORTED_OPERATION,
                "grammar-constrained decoding (json_schema / regex / ebnf / "
                "structural_tag) is not supported with beam search "
                "(num_beams > 1 or num_return_sequences > 1)",
            )

    def _reasoning_envelope_final_is_json(self) -> Optional[bool]:
        if self.config.structural_tag is None:
            return None
        structural_tag = load_json_field("structural_tag", self.config.structural_tag)
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
        if has_bounded_region(final_format):
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

    def _wrap_grammar_with_reasoning_envelope(
        self, constraint: GrammarConstraint
    ) -> None:
        final_format = constraint.final_format_node()
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
        self.config.structural_tag = dump_compact_json(envelope)
        self.config.json_schema = None
        self.config.regex = None
        self.config.ebnf = None
