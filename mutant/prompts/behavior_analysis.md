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
  "detected_domain": "string - inferred domain (e.g. e-commerce, healthcare)",
  "confidence": 0.95,
  "actors": ["string - people or systems involved"],
  "entities": ["string - key objects, identifiers"],
  "goals": ["string - what each actor is trying to achieve"],
  "constraints": ["string - rules and limits"],
  "assumptions": ["string - implicit assumptions"],
  "policies": ["string - business policies in play"],
  "tools": ["string - tools or APIs needed"],
  "risks": ["string - failure risks"],
  "likely_failure_modes": ["string - how a weak agent fails"],
  "ambiguities": ["string - unclear elements"]
}
```

Return ONLY the JSON. No explanation. No markdown fences.
