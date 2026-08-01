"""Standardized exception hierarchy for SciMCP."""

from __future__ import annotations
from typing import Any


class SciMCPError(Exception):
    """Base exception for all SciMCP errors."""

    def __init__(
        self,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": False,
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            },
        }


class InvalidInputError(SciMCPError):
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__("INVALID_INPUT", message, details)


class InvalidFileFormatError(SciMCPError):
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__("INVALID_FILE_FORMAT", message, details)


class FileNotFoundError(SciMCPError):
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__("FILE_NOT_FOUND", message, details)


class DependencyNotAvailableError(SciMCPError):
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__("DEPENDENCY_NOT_AVAILABLE", message, details)


class ExternalAPIError(SciMCPError):
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__("EXTERNAL_API_ERROR", message, details)


class ScientificValidationError(SciMCPError):
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__("SCIENTIFIC_VALIDATION_ERROR", message, details)


class ModelOutOfDomainError(SciMCPError):
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__("MODEL_OUT_OF_DOMAIN", message, details)


class CalculationFailedError(SciMCPError):
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__("CALCULATION_FAILED", message, details)
