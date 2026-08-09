from pathlib import Path

from ehk.common.settings import REPOSITORY_ROOT, register_path


def test_path_setting_uses_repository_relative_default(monkeypatch):
    setting = register_path("EHK_TEST_PATH", "outputs/test")
    monkeypatch.delenv("EHK_TEST_PATH", raising=False)
    assert setting.resolve() == REPOSITORY_ROOT / "outputs/test"


def test_path_setting_accepts_relative_environment_override(monkeypatch):
    setting = register_path("EHK_TEST_PATH", "outputs/test")
    monkeypatch.setenv("EHK_TEST_PATH", "custom/output")
    assert setting.resolve() == REPOSITORY_ROOT / "custom/output"


def test_path_setting_accepts_absolute_environment_override(monkeypatch, tmp_path):
    setting = register_path("EHK_TEST_PATH", "outputs/test")
    monkeypatch.setenv("EHK_TEST_PATH", str(tmp_path))
    assert setting.resolve() == Path(tmp_path).resolve()
