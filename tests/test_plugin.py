"""Plugin facade, coverage, crate pin, and loader contract tests."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest
from rextio.config.schema import PluginConfig, RextioConfig
from rextio.plugins.api import (
    PLUGIN_DIAGNOSTIC_CODE_PATTERN,
    CoverageDecl,
    CrateDependency,
    PluginType,
    RuleRecord,
)
from rextio.plugins.loader import load_plugin_registry
from rextio.targets.models import TargetSpec

from rextio_torch import __version__
from rextio_torch.plugin import PLUGIN_ID, REQUIRED_PLUGIN_API, RextioTorchPlugin, plugin
from rextio_torch.rules import COVERAGE, torch_rule_records

ROOT = Path(__file__).resolve().parents[1]


class FakeEntryPoint:
    name = PLUGIN_ID

    def load(self) -> Any:
        return plugin


def load_registry(enabled: tuple[str, ...] = (PLUGIN_ID,)):
    return load_plugin_registry(
        PluginConfig(enabled=enabled),
        TargetSpec(),
        entry_points=(FakeEntryPoint(),),
        full_config=RextioConfig(),
    )


def test_entry_point_factory_returns_plugin() -> None:
    obj = plugin()
    assert isinstance(obj, RextioTorchPlugin)
    assert obj.plugin_id == PLUGIN_ID
    assert obj.api_version == REQUIRED_PLUGIN_API == "1.3"
    assert __version__ == "0.1.2"


def test_core_loader_accepts_the_plugin() -> None:
    registry = load_registry()
    active = registry.active[0]
    assert active.id == PLUGIN_ID
    assert active.rules_provided is True
    assert active.lowering_provided is True
    assert active.api_version == "1.3"
    assert active.packages == ("torch",)
    assert __version__ in active.name
    assert registry.coverages[0].coverage == COVERAGE
    assert [record.id for record in registry.rule_records] == [
        record.id for record in torch_rule_records()
    ]


@pytest.mark.parametrize("host_api", ("1.3", "1.4", "1.5"))
def test_loader_negotiates_api_13_provider_without_artifact_capability(
    monkeypatch: pytest.MonkeyPatch, host_api: str
) -> None:
    """Core owns API compatibility; this provider remains host-extension-only."""
    import rextio.plugins.api as plugin_api

    monkeypatch.setattr(plugin_api, "PLUGIN_API_VERSION", host_api)
    registry = load_registry()

    assert registry.active[0].api_version == "1.3"
    assert getattr(registry.active[0], "artifact_capability_declared", False) is False
    assert not hasattr(plugin(), "artifact_capability")


@pytest.mark.parametrize(
    "host_api",
    ("1.2", "2.0", "1", "1.3.0", "not-a-version"),
)
def test_provider_registration_rejects_incompatible_host_api(
    monkeypatch: pytest.MonkeyPatch, host_api: str
) -> None:
    import rextio.plugins.api as plugin_api

    monkeypatch.setattr(plugin_api, "PLUGIN_API_VERSION", host_api)
    with pytest.raises(RuntimeError, match="compatible Rextio plugin host API"):
        RextioTorchPlugin().to_rextio_plugin()


def test_covers_alpha_aot_surface() -> None:
    coverage = plugin().covers()
    assert isinstance(coverage, CoverageDecl)
    assert coverage.packages == ("torch",)
    assert "torch.nn.functional" in coverage.modules
    assert "torch.nn.functional.linear" in coverage.symbols
    assert "torch.matmul" in coverage.symbols
    assert "torch.add" in coverage.symbols
    assert "torch.sub" in coverage.symbols
    assert "torch.mul" in coverage.symbols
    assert "torch.div" in coverage.symbols
    for operation in ("abs", "neg", "negative", "square", "exp", "log", "sqrt"):
        assert f"torch.{operation}" in coverage.symbols
        assert f"torch.Tensor.{operation}" in coverage.symbols
    assert "torch.Tensor.relu" in coverage.symbols
    assert "torch.Tensor.sigmoid" in coverage.symbols
    assert "torch.Tensor.tanh" in coverage.symbols
    assert "torch.Tensor.mean" in coverage.symbols
    assert "torch.Tensor.sum" in coverage.symbols
    assert "torch.Tensor.softmax" in coverage.symbols
    assert "torch.Tensor.argmax" in coverage.symbols


def test_rule_records_are_namespaced_and_well_formed() -> None:
    records = plugin().describe(RextioConfig())
    assert records
    codes: set[str] = set()
    for record in records:
        assert isinstance(record, RuleRecord)
        assert record.id.startswith("rextio-torch/")
        assert record.provider == "rextio-torch"
        if record.diagnostic_code is not None:
            match = PLUGIN_DIAGNOSTIC_CODE_PATTERN.match(record.diagnostic_code)
            assert match is not None
            assert match.group(1) == "TORCH"
            assert record.diagnostic_code not in codes
            codes.add(record.diagnostic_code)
    assert "RXTP-TORCH-001" in codes
    assert "RXTP-TORCH-002" in codes
    assert "RXTP-TORCH-003" in codes
    assert "RXTP-TORCH-010" in codes
    assert "RXTP-TORCH-014" in codes
    assert "RXTP-TORCH-015" in codes
    assert "RXTP-TORCH-016" in codes


def test_type_vocabulary_keys_and_boundary() -> None:
    types = plugin().type_vocabulary()
    assert {t.key for t in types} == {
        "rextio-torch/tensor-f32-cpu-2d",
        "rextio-torch/tensor-f32-cpu-1d",
        "rextio-torch/tensor-i64-cpu-1d",
    }
    for plugin_type in types:
        assert isinstance(plugin_type, PluginType)
        assert plugin_type.rust_type == "RxtTorchTensor"
        assert plugin_type.conversion is not None
        assert plugin_type.conversion.param_rust == "pyo3::Bound<'py, pyo3::types::PyAny>"
        assert plugin_type.conversion.return_expr == (
            "__rxttorch_materialize_tensor(py, {value})?"
        )
        assert plugin_type.helpers
        assert "struct RxtTorchTensor" in plugin_type.helpers[0]
        assert "shallow_clone" in plugin_type.helpers[0]
        assert "pyobject_unpack" in plugin_type.helpers[0]
        assert "pyobject_wrap" in plugin_type.helpers[0]
        assert "from_owned_ptr" in plugin_type.helpers[0]
    spellings = {a for t in types for a in t.annotations}
    assert spellings == {
        "rextio_torch.types.TensorF32Cpu2D",
        "rextio_torch.types.TensorF32Cpu1D",
        "rextio_torch.types.TensorI64Cpu1D",
    }


def test_crate_dependency_is_exact_tch_python_extension() -> None:
    deps = plugin().crate_dependencies()
    assert deps == (
        CrateDependency(name="tch", version="=0.24.0", features=("python-extension",)),
    )
    registry = load_registry()
    assert [(b.dependency.name, b.dependency.version, b.dependency.features) for b in registry.crate_dependencies] == [
        ("tch", "=0.24.0", ("python-extension",))
    ]


def test_public_alpha_release_candidate_metadata() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["build-system"]["requires"] == [
        "setuptools==82.0.1",
        "wheel==0.47.0",
    ]
    project = pyproject["project"]
    assert project["name"] == "rextio-torch"
    assert project["requires-python"] == ">=3.11,<3.12"
    assert project["description"].startswith("Public Alpha Rextio plugin")
    assert "Private :: Do Not Upload" not in project["classifiers"]
    assert "Development Status :: 3 - Alpha" in project["classifiers"]
    assert "Development Status :: 2 - Pre-Alpha" not in project["classifiers"]
    assert "Programming Language :: Python :: 3.11" in project["classifiers"]
    assert "Programming Language :: Python :: 3.12" not in project["classifiers"]
    dependencies = project["dependencies"]
    assert "rextio>=0.1.3,<0.2" in dependencies
    assert "torch==2.11.0" in dependencies
    assert all("git+" not in dep for dep in dependencies)
    assert "rextio-core-next" not in " ".join(dependencies)
    assert project["optional-dependencies"]["test"] == ["pytest==9.1.1"]
    assert project["optional-dependencies"]["dev"] == [
        "pytest==9.1.1",
        "ruff==0.15.22",
        "mypy==2.3.0",
        "build==1.5.0",
        "setuptools==81.0.0",
        "wheel==0.47.0",
        "twine==6.2.0",
        "check-wheel-contents==0.6.3",
    ]
