"""mutant/exceptions.py — V0.5 Custom Exceptions."""


class MutantError(Exception):
    """Base exception for all Mutant errors."""

    pass


class ProviderError(MutantError):
    """Raised when an LLM provider fails."""

    def __init__(
        self,
        message: str,
        stage: str,
        provider_name: str,
        original_error: Exception | None = None,
    ):
        super().__init__(
            f"Provider '{provider_name}' failed at stage '{stage}': {message}"
        )
        self.stage = stage
        self.provider_name = provider_name
        self.original_error = original_error


class ParseError(MutantError):
    """Raised when the LLM output could not be parsed into the expected JSON schema."""

    pass
