import json
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException

GrammarFieldName = Literal["json_schema", "regex", "ebnf", "structural_tag"]
GrammarValue = Union[str, Dict[str, Any]]

GRAMMAR_FIELD_NAMES: Tuple[GrammarFieldName, ...] = (
    "json_schema",
    "regex",
    "ebnf",
    "structural_tag",
)
COMPACT_JSON_SEPARATORS = (",", ":")


def dump_compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=COMPACT_JSON_SEPARATORS)


def load_json_field(name: str, value: GrammarValue) -> Any:
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError) as e:
        raise FtRuntimeException(
            ExceptionType.ERROR_INPUT_FORMAT_ERROR,
            f"{name} must be valid JSON: {str(e)}",
        )


def normalize_grammar_value(
    name: GrammarFieldName, value: GrammarValue
) -> GrammarValue:
    if name in ("json_schema", "structural_tag") and isinstance(value, dict):
        return dump_compact_json(value)
    return value


def has_bounded_region(node: Any) -> bool:
    if isinstance(node, dict):
        if (
            node.get("type") in ("any_text", "any_tokens")
            and node.get("max_tokens") is not None
        ):
            return True
        return any(has_bounded_region(value) for value in node.values())
    if isinstance(node, list):
        return any(has_bounded_region(item) for item in node)
    return False


@dataclass(frozen=True)
class GrammarConstraint:
    """Canonical one-of grammar constraint before it is written to GenerateConfig fields."""

    name: GrammarFieldName
    value: GrammarValue

    @classmethod
    def from_response_format(
        cls, response_format: Any
    ) -> Optional["GrammarConstraint"]:
        if response_format is None or response_format.type == "text":
            return None
        if response_format.type == "json_schema":
            return cls("json_schema", response_format.json_schema.schema)
        if response_format.type == "json_object":
            return cls("json_schema", {"type": "object"})
        if response_format.type == "regex":
            return cls("regex", response_format.pattern)
        if response_format.type == "ebnf":
            return cls("ebnf", response_format.grammar)
        if response_format.type == "structural_tag":
            return cls("structural_tag", response_format.structural_tag)
        raise FtRuntimeException(
            ExceptionType.ERROR_INPUT_FORMAT_ERROR,
            f"unsupported response_format type {response_format.type}",
        )

    @classmethod
    def collect_from_config(cls, config: Any) -> List["GrammarConstraint"]:
        constraints = []
        for name in GRAMMAR_FIELD_NAMES:
            value = getattr(config, name)
            if value is not None:
                constraints.append(cls(name, value))
        return constraints

    def normalized(self) -> "GrammarConstraint":
        return GrammarConstraint(
            self.name, normalize_grammar_value(self.name, self.value)
        )

    def validate_not_empty(self) -> None:
        if isinstance(self.value, str) and not self.value.strip():
            raise FtRuntimeException(
                ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                f"{self.name} must not be empty",
            )

    def as_config_update(self) -> Dict[GrammarFieldName, Optional[GrammarValue]]:
        values: Dict[GrammarFieldName, Optional[GrammarValue]] = {
            "json_schema": None,
            "regex": None,
            "ebnf": None,
            "structural_tag": None,
        }
        normalized = self.normalized()
        values[normalized.name] = normalized.value
        return values

    def final_format_node(self) -> Dict[str, Any]:
        if self.name == "json_schema":
            schema: Any
            if self.value == "$$ANY$$":
                schema = True
            else:
                schema = load_json_field("json_schema", self.value)
            if not isinstance(schema, (dict, bool)):
                raise FtRuntimeException(
                    ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                    "json_schema must be a JSON object or boolean",
                )
            return {"type": "json_schema", "json_schema": schema, "style": "json"}
        if self.name == "regex":
            return {"type": "regex", "pattern": self.value}
        if self.name == "ebnf":
            return {"type": "grammar", "grammar": self.value}
        structural_tag = load_json_field("structural_tag", self.value)
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
        if has_bounded_region(format_node):
            raise FtRuntimeException(
                ExceptionType.UNSUPPORTED_OPERATION,
                "reasoning grammar cannot wrap a final structural_tag that "
                "already contains any_text/any_tokens max_tokens",
            )
        return format_node
