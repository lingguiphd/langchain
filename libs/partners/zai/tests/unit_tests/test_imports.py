"""Test `langchain_zai` public API surface."""

from langchain_zai import __all__

EXPECTED_ALL = ["__version__", "ChatZAI"]


def test_all_imports() -> None:
    """Verify that `__all__` exports match the expected public API."""
    assert sorted(EXPECTED_ALL) == sorted(__all__)
