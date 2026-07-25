"""Focused contract tests for the bounded CUDA E2 build-only candidate."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from rextio.config.schema import RextioConfig
from rextio.analyzer.project_scanner import analyze_project
from rextio.config.schema import PluginConfig
from rextio.devices import DeviceLoweringAuthorization, derive_device_requirements
from rextio.plugins.api import (
    ClaimLiteral,
    Claimed,
    ClaimSite,
    KeywordArg,
    LoweringContext,
    ReceiverMeta,
    Rejected,
)

from rextio_torch.claim.cuda import (
    CUDA_BIAS_ADD_RULE,
    CUDA_MATMUL_RULE,
    CUDA_MEAN_DIM1_RULE,
    CUDA_RELU_RULE,
)
from rextio_torch.diagnostics import (
    TENSOR_F32_CPU_2D,
    TENSOR_F32_CUDA0_1D,
    TENSOR_F32_CUDA0_2D,
)
from rextio_torch.lower.cuda import CUDA_CAPABILITY_ID, CUDA_PROVIDER_ID
from rextio_torch.plugin import plugin
from rextio_torch.plugin_types import CUDA_RUNTIME_REQUIREMENTS, plugin_type
from rextio_torch.rust_snippets import boundary_helpers
from rextio.plugins.loader import load_plugin_registry
from rextio.targets.models import TargetSpec
from scripts.build_cuda_candidate import _require_cuda_enabled_torch

PLUGIN = plugin()
CONFIG = RextioConfig()


class _EntryPoint:
    name = "rextio-torch"

    def load(self):
        return plugin


def _authorization(
    *,
    provider_id: str = CUDA_PROVIDER_ID,
    capability_id: str = CUDA_CAPABILITY_ID,
    runtime: str = "libtorch",
    features: tuple[str, ...] = ("inference", "no-grad"),
    layouts: tuple[str, ...] = ("strided",),
    memory_spaces: tuple[str, ...] = ("device",),
) -> DeviceLoweringAuthorization:
    return DeviceLoweringAuthorization(
        provider_id=provider_id,
        capability_id=capability_id,
        logical_device="cuda:0",
        backend="cuda",
        runtime=runtime,
        reuse_domain_runtime=True,
        features=features,
        layouts=layouts,
        memory_spaces=memory_spaces,
        artifact_profile_sha256="0" * 64,
    )


def _ctx(
    operands: tuple[str, ...] = (),
    *,
    receiver: str | None = None,
    authorization: DeviceLoweringAuthorization | None = None,
) -> LoweringContext:
    return LoweringContext(
        operands=operands,
        target_language="rust",
        fresh_name=lambda prefix: f"{prefix}_0",
        receiver=receiver,
        device_authorization=authorization,
    )


def _keywords(*, keepdim: bool | None = None) -> tuple[KeywordArg, ...]:
    values = [
        KeywordArg(
            name="dim",
            arg_type="int",
            literal=ClaimLiteral(is_literal=True, value=1),
        )
    ]
    if keepdim is not None:
        values.append(
            KeywordArg(
                name="keepdim",
                arg_type="bool",
                literal=ClaimLiteral(is_literal=True, value=keepdim),
            )
        )
    return tuple(values)


def test_cuda_plugin_types_carry_exact_api_16_device_metadata() -> None:
    metadata_values = []
    for key, rank in (
        (TENSOR_F32_CUDA0_2D, 2),
        (TENSOR_F32_CUDA0_1D, 1),
    ):
        item = plugin_type(key)
        metadata = item.device_value_metadata
        assert metadata is not None
        metadata_values.append(metadata)
        assert metadata.logical_device == "gpu:0"
        assert metadata.backend == "cuda"
        assert metadata.dtype == "float32"
        assert metadata.rank == rank
        assert metadata.layout == "strided"
        assert metadata.runtime == "libtorch"
        assert metadata.runtime_version == "2.11.0"
        assert metadata.reuse_domain_runtime is True
        assert metadata.features == ("inference", "no-grad")
        assert metadata.memory_spaces == ("device",)
        assert metadata.runtime_requirements == CUDA_RUNTIME_REQUIREMENTS
        assert item.conversion is not None
        assert f"__rxttorch_extract_f32_cuda0_{rank}d" in item.conversion.param_expr
        assert f"__rxttorch_materialize_f32_cuda0_{rank}d" in item.conversion.return_expr
    requirements = derive_device_requirements(tuple(metadata_values))
    assert len(requirements) == 1
    requirement = requirements[0]
    assert requirement.logical_device == "gpu:0"
    assert requirement.backend == "cuda"
    assert requirement.runtime == "libtorch"
    assert requirement.reuse_domain_runtime is True
    assert requirement.features == ("inference", "no-grad")
    assert requirement.layouts == ("strided",)
    assert requirement.memory_spaces == ("device",)


def test_claims_only_the_frozen_cuda_vertical_slice() -> None:
    matmul = ClaimSite(
        kind="binop",
        target="@",
        operand_types=(TENSOR_F32_CUDA0_2D, TENSOR_F32_CUDA0_2D),
        file_path="",
        line=0,
        column=0,
    )
    assert PLUGIN.claim(matmul, CONFIG) == Claimed(
        rule_id=CUDA_MATMUL_RULE, result_type=TENSOR_F32_CUDA0_2D
    )

    bias_add = ClaimSite(
        kind="binop",
        target="+",
        operand_types=(TENSOR_F32_CUDA0_2D, TENSOR_F32_CUDA0_1D),
        file_path="",
        line=0,
        column=0,
    )
    assert PLUGIN.claim(bias_add, CONFIG) == Claimed(
        rule_id=CUDA_BIAS_ADD_RULE, result_type=TENSOR_F32_CUDA0_2D
    )

    relu = ClaimSite(
        kind="call",
        target="tensor.relu",
        operand_types=(),
        file_path="",
        line=0,
        column=0,
        receiver=ReceiverMeta(
            arg_type=TENSOR_F32_CUDA0_2D, expr_kind="name", is_safe=True
        ),
    )
    assert PLUGIN.claim(relu, CONFIG) == Claimed(
        rule_id=CUDA_RELU_RULE, result_type=TENSOR_F32_CUDA0_2D
    )

    mean = ClaimSite(
        kind="call",
        target="tensor.mean",
        operand_types=(),
        file_path="",
        line=0,
        column=0,
        receiver=ReceiverMeta(
            arg_type=TENSOR_F32_CUDA0_2D, expr_kind="name", is_safe=True
        ),
        keywords=_keywords(),
    )
    assert PLUGIN.claim(mean, CONFIG) == Claimed(
        rule_id=CUDA_MEAN_DIM1_RULE, result_type=TENSOR_F32_CUDA0_1D
    )


@pytest.mark.parametrize(
    ("target", "types"),
    (
        ("+", (TENSOR_F32_CUDA0_1D, TENSOR_F32_CUDA0_2D)),
        ("+", (TENSOR_F32_CUDA0_2D, TENSOR_F32_CUDA0_2D)),
        ("@", (TENSOR_F32_CUDA0_2D, TENSOR_F32_CUDA0_1D)),
        ("@", (TENSOR_F32_CUDA0_2D, TENSOR_F32_CPU_2D)),
        ("*", (TENSOR_F32_CUDA0_2D, TENSOR_F32_CUDA0_2D)),
    ),
)
def test_cuda_does_not_inherit_the_broader_cpu_binop_matrix(
    target: str, types: tuple[str, str]
) -> None:
    site = ClaimSite(
        kind="binop",
        target=target,
        operand_types=types,
        file_path="",
        line=0,
        column=0,
    )
    assert isinstance(PLUGIN.claim(site, CONFIG), Rejected)


@pytest.mark.parametrize(
    ("target", "operand_types", "receiver_type", "keywords"),
    (
        ("torch.matmul", (TENSOR_F32_CUDA0_2D, TENSOR_F32_CUDA0_2D), None, ()),
        ("torch.relu", (TENSOR_F32_CUDA0_2D,), None, ()),
        ("tensor.sum", (), TENSOR_F32_CUDA0_2D, _keywords()),
        ("tensor.sigmoid", (), TENSOR_F32_CUDA0_2D, ()),
        ("tensor.tanh", (), TENSOR_F32_CUDA0_2D, ()),
    ),
)
def test_cuda_functional_and_out_of_slice_calls_are_rejected(
    target: str,
    operand_types: tuple[str, ...],
    receiver_type: str | None,
    keywords: tuple[KeywordArg, ...],
) -> None:
    receiver = (
        ReceiverMeta(arg_type=receiver_type, expr_kind="name", is_safe=True)
        if receiver_type is not None
        else None
    )
    site = ClaimSite(
        kind="call",
        target=target,
        operand_types=operand_types,
        file_path="",
        line=0,
        column=0,
        receiver=receiver,
        keywords=keywords,
    )
    assert isinstance(PLUGIN.claim(site, CONFIG), Rejected)


def test_cuda_mean_rejects_bool_disguised_as_integer_dimension() -> None:
    site = ClaimSite(
        kind="call",
        target="tensor.mean",
        operand_types=(),
        file_path="",
        line=0,
        column=0,
        receiver=ReceiverMeta(
            arg_type=TENSOR_F32_CUDA0_2D, expr_kind="name", is_safe=True
        ),
        keywords=(
            KeywordArg(
                name="dim",
                arg_type="int",
                literal=ClaimLiteral(is_literal=True, value=True),
            ),
        ),
    )
    assert isinstance(PLUGIN.claim(site, CONFIG), Rejected)


def test_cuda_mean_lower_rejects_forged_bool_dimension() -> None:
    forged = ClaimSite(
        kind="call",
        target="tensor.mean",
        operand_types=(),
        file_path="",
        line=0,
        column=0,
        receiver=ReceiverMeta(
            arg_type=TENSOR_F32_CUDA0_2D, expr_kind="name", is_safe=True
        ),
        keywords=(
            KeywordArg(
                name="dim",
                arg_type="int",
                literal=ClaimLiteral(is_literal=True, value=True),
            ),
        ),
        rule_id=CUDA_MEAN_DIM1_RULE,
        result_type=TENSOR_F32_CUDA0_1D,
    )
    with pytest.raises(ValueError, match="metadata changed"):
        PLUGIN.lower(
            forged,
            _ctx(receiver="activated", authorization=_authorization()),
        )


def test_cuda_lower_requires_exact_provider_and_capability_authorization() -> None:
    claimed = ClaimSite(
        kind="binop",
        target="@",
        operand_types=(TENSOR_F32_CUDA0_2D, TENSOR_F32_CUDA0_2D),
        file_path="",
        line=0,
        column=0,
        rule_id=CUDA_MATMUL_RULE,
        result_type=TENSOR_F32_CUDA0_2D,
    )
    with pytest.raises(ValueError, match="exact authorization"):
        PLUGIN.lower(claimed, _ctx(("x", "w")))
    with pytest.raises(ValueError, match="exact authorization"):
        PLUGIN.lower(
            claimed,
            _ctx(
                ("x", "w"),
                authorization=_authorization(provider_id="other-cuda"),
            ),
        )
    for authorization in (
        _authorization(runtime="other-runtime"),
        _authorization(features=("inference",)),
        _authorization(layouts=("contiguous",)),
        _authorization(memory_spaces=("host",)),
    ):
        with pytest.raises(ValueError, match="exact authorization"):
            PLUGIN.lower(claimed, _ctx(("x", "w"), authorization=authorization))
    with pytest.raises(ValueError, match="exact authorization"):
        PLUGIN.lower(
            claimed,
            _ctx(
                ("x", "w"),
                authorization=_authorization(capability_id="cuda-linux-x86_64"),
            ),
        )

    lowered = PLUGIN.lower(
        claimed,
        _ctx(("x", "w"), authorization=_authorization()),
    )
    assert lowered.rust == "__rxttorch_matmul(&x, &w)?"
    assert "f_matmul" in "\n".join(lowered.helpers)

    forged_result = ClaimSite(
        kind="binop",
        target="@",
        operand_types=(TENSOR_F32_CUDA0_2D, TENSOR_F32_CUDA0_2D),
        file_path="",
        line=0,
        column=0,
        rule_id=CUDA_MATMUL_RULE,
        result_type="rextio-torch/forged-cuda-result",
    )
    with pytest.raises(ValueError, match="exact authorization"):
        PLUGIN.lower(
            forged_result,
            _ctx(("x", "w"), authorization=_authorization()),
        )


def test_cuda_lower_revalidates_add_relu_and_mean() -> None:
    authorization = _authorization()
    add = ClaimSite(
        kind="binop",
        target="+",
        operand_types=(TENSOR_F32_CUDA0_2D, TENSOR_F32_CUDA0_1D),
        file_path="",
        line=0,
        column=0,
        rule_id=CUDA_BIAS_ADD_RULE,
        result_type=TENSOR_F32_CUDA0_2D,
    )
    assert PLUGIN.lower(
        add, _ctx(("m", "b"), authorization=authorization)
    ).rust == "__rxttorch_add(&m, &b)?"

    receiver = ReceiverMeta(
        arg_type=TENSOR_F32_CUDA0_2D, expr_kind="name", is_safe=True
    )
    relu = ClaimSite(
        kind="call",
        target="tensor.relu",
        operand_types=(),
        file_path="",
        line=0,
        column=0,
        receiver=receiver,
        rule_id=CUDA_RELU_RULE,
        result_type=TENSOR_F32_CUDA0_2D,
    )
    assert PLUGIN.lower(
        relu, _ctx(receiver="hidden", authorization=authorization)
    ).rust == "__rxttorch_relu(&hidden)?"

    mean = ClaimSite(
        kind="call",
        target="tensor.mean",
        operand_types=(),
        file_path="",
        line=0,
        column=0,
        receiver=receiver,
        keywords=_keywords(keepdim=False),
        rule_id=CUDA_MEAN_DIM1_RULE,
        result_type=TENSOR_F32_CUDA0_1D,
    )
    assert PLUGIN.lower(
        mean, _ctx(receiver="activated", authorization=authorization)
    ).rust == "__rxttorch_mean_dim1_keepdim_false(&activated)?"


def test_cuda_boundary_is_zero_copy_and_rank_specific() -> None:
    helper = boundary_helpers()
    assert "tch::Device::Cuda(0)" in helper
    assert "tch::Kind::Float" in helper
    assert "__rxttorch_extract_f32_cuda0_2d" in helper
    assert "__rxttorch_extract_f32_cuda0_1d" in helper
    assert "__rxttorch_materialize_f32_cuda0_2d" in helper
    assert "__rxttorch_materialize_f32_cuda0_1d" in helper
    assert "pyobject_unpack" in helper
    assert "pyobject_wrap" in helper
    assert ".to_device(" not in helper
    assert ".to(" not in helper
    assert ".cpu(" not in helper
    assert 'getattr("layout")' in helper
    assert '"torch.strided"' in helper
    assert "tensor.is_sparse()" in helper
    assert "tensor.is_mkldnn()" in helper
    assert helper.count("__rxttorch_require_python_strided(") == 3
    assert helper.count("__rxttorch_require_native_strided(") == 3

    extraction = helper[
        helper.index("fn __rxttorch_extract_f32_cuda0(") :
        helper.index("fn __rxttorch_extract_f32_cuda0_2d(")
    ]
    assert extraction.index("__rxttorch_require_python_strided(value)?") < extraction.index(
        "tch::Device::Cuda(0)"
    )
    assert extraction.index("__rxttorch_require_native_strided(&tensor)?") < extraction.index(
        "tch::Device::Cuda(0)"
    )

    materialization = helper[
        helper.index("fn __rxttorch_materialize_f32_cuda0(") :
        helper.index("fn __rxttorch_materialize_f32_cuda0_2d(")
    ]
    assert materialization.index("__rxttorch_require_native_strided(&value.0)?") < (
        materialization.index("pyobject_wrap")
    )
    assert materialization.index("pyobject_wrap") < materialization.index(
        "__rxttorch_require_python_strided(&wrapped)?"
    )


def test_cuda_build_script_rejects_cpu_only_libtorch() -> None:
    with pytest.raises(SystemExit, match="CUDA-enabled PyTorch"):
        _require_cuda_enabled_torch(SimpleNamespace(version=SimpleNamespace(cuda=None)))
    with pytest.raises(SystemExit, match="torch.version.cuda"):
        _require_cuda_enabled_torch(SimpleNamespace(version=SimpleNamespace(cuda="")))
    assert (
        _require_cuda_enabled_torch(
            SimpleNamespace(version=SimpleNamespace(cuda="12.8"))
        )
        == "12.8"
    )


def test_cuda_diagnostic_constants_are_public() -> None:
    import rextio_torch.diagnostics as diagnostics

    assert {
        "DIAGNOSTIC_CUDA_MATMUL",
        "DIAGNOSTIC_CUDA_BIAS_ADD",
        "DIAGNOSTIC_CUDA_RELU",
        "DIAGNOSTIC_CUDA_MEAN",
        "DIAGNOSTIC_CUDA_E2",
    } <= set(diagnostics.__all__)


def test_analyzer_discovers_only_the_cuda_e2_chain(tmp_path) -> None:
    (tmp_path / "rextio.toml").write_text(
        '[rust]\nbuild_tool = "cargo"\n\n[plugins]\nenabled = ["rextio-torch"]\n',
        encoding="utf-8",
    )
    package = tmp_path / "src" / "cuda_app"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "kernels.py").write_text(
        """
from rextio_torch.types import TensorF32Cuda0_1D, TensorF32Cuda0_2D


def inference(
    x: TensorF32Cuda0_2D,
    weight: TensorF32Cuda0_2D,
    bias: TensorF32Cuda0_1D,
) -> TensorF32Cuda0_1D:
    hidden = x @ weight
    biased = hidden + bias
    activated = biased.relu()
    return activated.mean(dim=1)
""",
        encoding="utf-8",
    )
    registry = load_plugin_registry(
        PluginConfig(enabled=("rextio-torch",)),
        TargetSpec(),
        entry_points=(_EntryPoint(),),
        full_config=CONFIG,
    )
    analysis = analyze_project(
        tmp_path,
        active_plugins=registry.active,
        plugin_registry=registry,
        plugin_config=CONFIG,
    )
    function = analysis.modules[1].functions[0]
    assert function.accepted is True
    assert [claim.rule_id for claim in function.plugin_claims] == [
        CUDA_MATMUL_RULE,
        CUDA_BIAS_ADD_RULE,
        CUDA_RELU_RULE,
        CUDA_MEAN_DIM1_RULE,
    ]
