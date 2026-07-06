"""Shared fixtures for backend tests."""

import pytest

from backend.common.config import PipelineSettings, get_settings


@pytest.fixture
def pipeline_config() -> PipelineSettings:
    return get_settings().pipeline
