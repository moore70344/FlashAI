"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture
def sample_training_data():
    """Sample training data for tests."""
    return {
        "examples": [
            {
                "query": "What is 2+2?",
                "answer": "4",
                "negative_samples": ["3", "5", "22"],
            },
            {
                "query": "What color is the sky?",
                "answer": "blue",
                "negative_samples": ["green", "red", "purple"],
            },
        ]
    }


@pytest.fixture
def sample_context():
    """Sample context for reasoning."""
    return {
        "domain": "mathematics",
        "difficulty": "easy",
        "user_preferences": {
            "verbosity": "concise",
        },
    }
