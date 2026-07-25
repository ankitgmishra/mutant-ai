# Security Policy

## Supported Versions

Currently, only the latest minor release is actively supported with security updates.

| Version | Supported          |
| ------- | ------------------ |
| >= 0.7.x | :white_check_mark: |
| < 0.7.x  | :x:                |

## Reporting a Vulnerability

We take the security of Mutant seriously. If you discover a vulnerability, please report it immediately.

**Do NOT report security vulnerabilities via public GitHub issues.**

Instead, please email **ankit@mutant-ai.dev** with a description of the vulnerability and the steps to reproduce it. 

### What to include in your report:
- A description of the vulnerability and its potential impact.
- Step-by-step instructions to reproduce the issue.
- The version of Mutant you are using.
- Any relevant context regarding the LLM provider configurations being used.

### Response Timeline
- We will acknowledge receipt of your vulnerability report within 48 hours.
- We aim to triage and confirm the vulnerability within 5 business days.
- A patch will be developed and released as quickly as possible depending on the severity.

## LLM API Key Security
Mutant requires API keys to interact with external providers (OpenAI, Anthropic, Gemini). 
- Mutant **does not** collect, store, or transmit your API keys to any third-party telemetry services.
- Keys are passed directly to the official SDKs (e.g., `openai`, `google-genai`).
- Please ensure you use environment variables (`.env`) to load your keys safely and never hardcode them in your scripts or commit them to version control.
