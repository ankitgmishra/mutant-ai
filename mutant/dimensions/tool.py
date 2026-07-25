"""Tool failure mutation dimensions."""

from mutant.core.mutation import MutationCategory, MutationSeverity
from mutant.dimensions.base import MutationDimension


class ToolTimeoutDimension(MutationDimension):
    id = "tool.timeout"
    name = "Tool Timeout"
    description = "A downstream tool or API the agent depends on times out."
    category = MutationCategory.TOOL
    severity = MutationSeverity.HIGH

    def get_mutation_instructions(self) -> str:
        return (
            "Annotate the scenario to simulate a tool timeout condition. The scenario "
            "description should include a system-level note (clearly marked as a "
            "[SYSTEM CONDITION]) indicating that a key downstream tool or API has "
            "timed out and is unavailable. The agent must handle this gracefully: "
            "inform the user, offer alternatives, and not expose internal system "
            "details. The underlying user request from the scenario remains valid."
        )


class EmptyToolResponseDimension(MutationDimension):
    id = "tool.empty_response"
    name = "Empty Tool Response"
    description = "A tool returns an empty or null response with no data."
    category = MutationCategory.TOOL
    severity = MutationSeverity.HIGH

    def get_mutation_instructions(self) -> str:
        return (
            "Annotate the scenario to simulate a tool returning an empty result. "
            "Include a [SYSTEM CONDITION] note indicating the relevant tool (lookup, "
            "database query, API call) returned no data — empty array, null, or "
            "zero results. The agent must handle this gracefully without assuming "
            "the worst case, and should explore alternatives or escalation paths."
        )


class InvalidJsonToolResponseDimension(MutationDimension):
    id = "tool.invalid_json"
    name = "Malformed Tool Response"
    description = "A tool returns malformed, unparseable data."
    category = MutationCategory.TOOL
    severity = MutationSeverity.HIGH

    def get_mutation_instructions(self) -> str:
        return (
            "Annotate the scenario to simulate a tool returning malformed data. "
            "Include a [SYSTEM CONDITION] note indicating the tool returned "
            "unparseable JSON, an unexpected content type, or corrupted data. "
            "The agent must handle parse failures gracefully, communicate "
            "appropriately to the user, and pursue fallback resolution paths."
        )


class ToolPermissionDeniedDimension(MutationDimension):
    id = "tool.permission_denied"
    name = "Tool Permission Denied"
    description = "The agent lacks permission to execute the required tool action."
    category = MutationCategory.TOOL
    severity = MutationSeverity.HIGH

    def get_mutation_instructions(self) -> str:
        return (
            "Annotate the scenario to simulate an authorization failure. Include "
            "a [SYSTEM CONDITION] note indicating the agent received a permission "
            "denied error (HTTP 403 or equivalent) when attempting to execute a "
            "required action. The agent must acknowledge its limitations without "
            "exposing internal system details, and provide a constructive path "
            "forward for the user."
        )


class WrongSchemaToolResponseDimension(MutationDimension):
    id = "tool.wrong_schema"
    name = "Schema Mismatch Tool Response"
    description = "A tool returns data with an unexpected field structure."
    category = MutationCategory.TOOL
    severity = MutationSeverity.MEDIUM

    def get_mutation_instructions(self) -> str:
        return (
            "Annotate the scenario to simulate a tool returning data with unexpected "
            "field names or structure. Include a [SYSTEM CONDITION] note indicating "
            "specific field name mismatches or missing required fields. The agent "
            "must handle schema drift gracefully — either inferring the correct "
            "mapping or escalating appropriately."
        )
