# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import tempfile
from collections.abc import Generator
from unittest.mock import patch

import pytest
from sqlalchemy import inspect

from pyrit.common.path import DATASETS_PATH
from pyrit.memory.central_memory import CentralMemory
from pyrit.memory.sqlite_memory import SQLiteMemory
from pyrit.models import SeedDataset

# This limits retries and speeds up execution
os.environ["CUSTOM_RESULT_RETRY_MAX_NUM_ATTEMPTS"] = "5"
os.environ["RETRY_MAX_NUM_ATTEMPTS"] = "2"
os.environ["RETRY_WAIT_MIN_SECONDS"] = "0"
os.environ["RETRY_WAIT_MAX_SECONDS"] = "1"


@pytest.fixture
def garak_api_key_service_patterns() -> dict[str, str | None]:
    dataset = SeedDataset.from_yaml_file(
        DATASETS_PATH / "seed_datasets" / "local" / "garak" / "api_key_service_patterns.prompt"
    )
    patterns: dict[str, str | None] = {}
    for prompt in dataset.prompts:
        pattern_name = prompt.metadata["pattern_name"]
        assert pattern_name is None or isinstance(pattern_name, str)
        assert prompt.value not in patterns, f"Duplicate service: {prompt.value}"
        patterns[prompt.value] = pattern_name
    return patterns


@pytest.fixture
def sqlite_instance() -> Generator[SQLiteMemory, None, None]:
    # Create an in-memory SQLite engine
    sqlite_memory = SQLiteMemory(db_path=":memory:")
    temp_dir = tempfile.TemporaryDirectory()
    sqlite_memory.results_path = temp_dir.name

    sqlite_memory.disable_embedding()

    # Reset the database to ensure a clean state
    sqlite_memory.reset_database()
    inspector = inspect(sqlite_memory.engine)

    # Verify that tables are created as expected
    assert "PromptMemoryEntries" in inspector.get_table_names(), "PromptMemoryEntries table not created."
    assert "EmbeddingData" in inspector.get_table_names(), "EmbeddingData table not created."
    assert "ScoreEntries" in inspector.get_table_names(), "ScoreEntries table not created."
    assert "SeedPromptEntries" in inspector.get_table_names(), "SeedPromptEntries table not created."

    CentralMemory.set_memory_instance(sqlite_memory)
    yield sqlite_memory
    temp_dir.cleanup()
    sqlite_memory.dispose_engine()


@pytest.fixture()
def patch_central_database(sqlite_instance):
    """Fixture to mock CentralMemory.get_memory_instance"""
    with patch.object(CentralMemory, "get_memory_instance", return_value=sqlite_instance) as sqlite_memory:
        yield sqlite_memory
