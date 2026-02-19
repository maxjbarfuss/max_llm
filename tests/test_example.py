"""Tests for example module."""

from src.example import ExampleClass


def test_example_class():
    """Test ExampleClass."""
    obj = ExampleClass("test")
    assert obj.get_name() == "test"
