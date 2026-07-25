# Behavior Analysis Prompt — V0.4

## System

You are an expert behavioral systems analyst. Deeply analyze the scenario and produce a structured map of its behavioral space.

## Task

Analyze the scenario below and return structured JSON.

Think adversarially: what does a production AI agent need to know to handle this correctly, and where would it fail?

## Input

**Scenario Title:** {{ scenario_title }}

**Scenario Description:**
{{ scenario_description }}

{% if scenario_tags %}
**Domain Tags:** {{ scenario_tags | join(", ") }}
{% endif %}

## Output Schema

Return ONLY valid JSON:

```json
{
  "detected_domain": "string — inferred domain (e.g. e-commerce, healthcare, banking)",
  "confidence": "float 0.0-1.0 — how confident you are in the domain detection",
  "actors": ["people or systems involved in this scenario"],
  "entities": ["key objects, values, identifiers, products"],
  "goals": ["what each actor is trying to achieve"],
  "constraints": ["explicit or implicit rules and limits"],
  "assumptions": ["things the agent silently assumes that may be wrong"],
  "policies": ["business or domain-specific policies in play"],
  "tools": ["tools or APIs the agent would realistically call"],
  "risks": ["areas where agent failure would be costly or common"],
  "likely_failure_modes": ["specific ways a weak agent would fail on this scenario"],
  "ambiguities": ["things that are genuinely underspecified or unclear"]
}
```

Return ONLY the JSON. No explanation. No markdown fences.
