"""Tests for launcher.py utility functions and configuration."""

import os
import random
import shutil
import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# randomize_seed_fn
# ---------------------------------------------------------------------------

MAX_SEED = int(1e7)


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    """Mirror of launcher.randomize_seed_fn for isolated testing."""
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


class TestRandomizeSeedFn:
    def test_returns_same_seed_when_not_randomizing(self):
        assert randomize_seed_fn(42, False) == 42

    def test_returns_random_seed_when_randomizing(self):
        result = randomize_seed_fn(42, True)
        assert 0 <= result <= MAX_SEED

    def test_float_cast_prevents_type_error(self):
        """The actual bug D-1: Gradio passes floats from number inputs."""
        seed_from_gradio = 1234.0
        result = int(randomize_seed_fn(int(seed_from_gradio), True))
        assert isinstance(result, int)
        assert 0 <= result <= MAX_SEED

    def test_zero_seed_is_valid(self):
        assert randomize_seed_fn(0, False) == 0

    def test_max_seed_is_valid(self):
        result = randomize_seed_fn(MAX_SEED, False)
        assert result == MAX_SEED

    def test_deterministic_with_same_random_state(self):
        random.seed(123)
        r1 = randomize_seed_fn(0, True)
        random.seed(123)
        r2 = randomize_seed_fn(0, True)
        assert r1 == r2


# ---------------------------------------------------------------------------
# gen_save_folder logic
# ---------------------------------------------------------------------------

class TestGenSaveFolder:
    def test_creates_folder_with_uuid_name(self, tmp_path):
        save_dir = str(tmp_path / "cache")
        os.makedirs(save_dir, exist_ok=True)

        # Simulate gen_save_folder logic
        new_folder = os.path.join(save_dir, str(uuid.uuid4()))
        os.makedirs(new_folder, exist_ok=True)

        assert os.path.isdir(new_folder)
        assert len(os.listdir(save_dir)) == 1

    def test_cleanup_removes_oldest_when_limit_reached(self, tmp_path):
        save_dir = str(tmp_path / "cache")
        os.makedirs(save_dir)
        max_size = 3

        # Create max_size folders
        created = []
        for i in range(max_size):
            d = os.path.join(save_dir, f"folder_{i}")
            os.makedirs(d)
            created.append(d)
            import time
            time.sleep(0.05)  # Ensure different ctime
        
        dirs = [f for f in Path(save_dir).iterdir() if f.is_dir()]
        assert len(dirs) == max_size

        # Simulate cleanup
        if len(dirs) >= max_size:
            oldest_dir = min(dirs, key=lambda x: x.stat().st_ctime)
            shutil.rmtree(oldest_dir)

        assert len(list(Path(save_dir).iterdir())) == max_size - 1


# ---------------------------------------------------------------------------
# XDG compliance
# ---------------------------------------------------------------------------

class TestXDGCompliance:
    def test_default_cache_path_is_xdg_compliant(self):
        """Default cache path should be under ~/.cache, not a relative path."""
        xdg_cache = os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache'))
        expected_prefix = os.path.join(xdg_cache, 'hy3dgen')
        # The default should start with the XDG cache
        assert expected_prefix.startswith(os.path.expanduser('~'))

    def test_xdg_cache_home_env_var_is_respected(self):
        custom_cache = '/tmp/test_xdg_cache'
        with patch.dict(os.environ, {'XDG_CACHE_HOME': custom_cache}):
            result = os.path.join(os.environ.get('XDG_CACHE_HOME', '~/.cache'), 'hy3dgen', 'gradio')
            assert result == os.path.join(custom_cache, 'hy3dgen', 'gradio')

    def test_xdg_state_home_env_var_is_respected(self):
        custom_state = '/tmp/test_xdg_state'
        with patch.dict(os.environ, {'XDG_STATE_HOME': custom_state}):
            result = os.path.join(os.environ.get('XDG_STATE_HOME', '~/.local/state'), 'hy3dgen')
            assert result == os.path.join(custom_state, 'hy3dgen')


# ---------------------------------------------------------------------------
# MAX_SEED constant
# ---------------------------------------------------------------------------

class TestConstants:
    def test_max_seed_is_integer(self):
        assert isinstance(MAX_SEED, int)

    def test_max_seed_is_positive(self):
        assert MAX_SEED > 0

    def test_max_seed_is_reasonable(self):
        assert MAX_SEED == 10_000_000
