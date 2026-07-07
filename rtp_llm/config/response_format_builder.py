from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any, Dict, List, Optional, Union

from pydantic import ValidationError

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.grammar_constraint import (
    GrammarConstraint,
    dump_compact_json,
    has_bounded_region,
    load_json_field,
    normalize_grammar_value,
)
from rtp_llm.config.response_format import parse_response_format

DEFAULT_THINK_END_TAG = "</think>\n\n"


@dataclass(frozen=True)
class ReasoningFormat:
    """Server/model resolved reasoning envelope format used for grammar wrapping."""

    tag_end: Union[str, List[str], Dict[str, Any]] = DEFAULT_THINK_END_TAG
    suffix: str = ""

    @classmethod
    def from_generate_env_config(cls, generate_env_config: Any) -> "ReasoningFormat":
        raw_tag = generate_env_config.think_end_tag or DEFAULT_THINK_END_TAG
        tag = str(raw_tag).encode("utf-8").decode("unicode_escape")
        raw_token_id = generate_env_config.think_end_token_id
        token_id = -1 if raw_token_id is None else int(raw_token_id)
        if token_id != -1:
            return cls(tag_end={"type": "token", "token": int(token_id)})
        return cls(tag_end=tag)

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

        if self._existing_reasoning_envelope_final_format() is not None:
            return

        if constraint is not None:
            self._wrap_grammar_with_reasoning_envelope(constraint)

    @classmethod
    def grammar_terminate_without_stop_token(cls, config: Any) -> bool:
        if config.json_schema is not None:
            return True
        final_format = cls(config)._existing_reasoning_envelope_final_format()
        return (
            final_format is not None and final_format.get("type") == "json_schema"
        )

    def _project_response_format_to_grammar_fields(self) -> None:
        """Project response_format onto typed fields and clear it; rf wins over stale extra_configs grammar."""
        raw_response_format = self.config.response_format
        if raw_response_format is None:
            return

        try:
            rf = parse_response_format(raw_response_format)
        except (JSONDecodeError, ValidationError, TypeError) as e:
            raise FtRuntimeException(
                ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                f"response_format invalid: {str(e)}",
            )

        self.config.response_format = rf
        if rf is None:
            return

        constraint = GrammarConstraint.from_response_format(rf)
        self.config.response_format = None
        self.config.json_schema = None
        self.config.regex = None
        self.config.ebnf = None
        self.config.structural_tag = None

        if constraint is None:
            return

        normalized = constraint.normalized()
        if normalized.name == "json_schema":
            self.config.json_schema = normalized.value
        elif normalized.name == "regex":
            self.config.regex = normalized.value
        elif normalized.name == "ebnf":
            self.config.ebnf = normalized.value
        elif normalized.name == "structural_tag":
            self.config.structural_tag = normalized.value
        else:
            raise FtRuntimeException(
                ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                f"unsupported grammar field {normalized.name}",
            )

    def _resolve_grammar_constraint(self) -> Optional[GrammarConstraint]:
        self._project_response_format_to_grammar_fields()
        if self.config.json_schema is not None:
            self.config.json_schema = normalize_grammar_value(
                "json_schema", self.config.json_schema
            )
        if self.config.regex is not None:
            self.config.regex = normalize_grammar_value("regex", self.config.regex)
        if self.config.ebnf is not None:
            self.config.ebnf = normalize_grammar_value("ebnf", self.config.ebnf)
        if self.config.structural_tag is not None:
            self.config.structural_tag = normalize_grammar_value(
                "structural_tag", self.config.structural_tag
            )
        constraints = GrammarConstraint.collect_from_config(self.config)
        self._validate_grammar_constraints(constraints)
        if not constraints:
            return None
        return constraints[0]

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

    def _existing_reasoning_envelope_final_format(self) -> Optional[Dict[str, Any]]:
        """Return final output format if structural_tag is already reasoning-wrapped."""
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
        if len(elements) not in (2, 3):
            return None

        reasoning_prefix = elements[0]
        if not isinstance(reasoning_prefix, dict):
            return None
        content = reasoning_prefix.get("content")
        if (
            reasoning_prefix.get("type") != "tag"
            or reasoning_prefix.get("begin") != ""
            or not isinstance(content, dict)
            or content.get("type") != "any_text"
            or content.get("max_tokens") is None
            or "end" not in reasoning_prefix
        ):
            return None

        if len(elements) == 3:
            suffix = elements[1]
            if not isinstance(suffix, dict) or suffix.get("type") != "const_string":
                return None
            final_format = elements[2]
        else:
            final_format = elements[1]

        if not isinstance(final_format, dict) or has_bounded_region(final_format):
            return None
        return final_format

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
