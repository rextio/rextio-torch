"""The rextio-torch plugin object and entry-point factory.

Implements plugin API 1.7: describe/covers, device-aware annotation vocabulary,
claim/lower, and the exact ``tch =0.24.0`` crate pin with
``python-extension``. API 1.7 adds one fail-closed function-scope
``tch::no_grad_guard()`` for eligible native PyO3 functions. This module never
imports torch; user-facing types are also import-free.

Import-time contract: this module (and therefore the package root and
:mod:`rextio_torch.types`) must load without analyzer/config/plugin modules
from core. Core types are imported lazily inside methods that only run under a
full analyzer/plugin host.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rextio_torch.__about__ import __version__

if TYPE_CHECKING:
    from rextio.config.schema import RextioConfig
    from rextio.plugins.api import (
        ClaimResult,
        ClaimSite,
        CoverageDecl,
        CrateDependency,
        LoweredExpr,
        LoweringContext,
        PluginFunctionScopeContext,
        PluginFunctionScopeGuard,
        PluginType,
        RuleRecord,
    )
    from rextio.plugins.models import RextioPlugin

PLUGIN_ID = "rextio-torch"
REQUIRED_PLUGIN_API = "1.7"

__all__ = ["PLUGIN_ID", "REQUIRED_PLUGIN_API", "RextioTorchPlugin", "plugin"]


def _require_compatible_host_api() -> None:
    from rextio.plugins.api import PLUGIN_API_VERSION

    parts = (
        PLUGIN_API_VERSION.split(".")
        if isinstance(PLUGIN_API_VERSION, str)
        else []
    )
    compatible = (
        len(parts) == 2
        and all(part.isdecimal() for part in parts)
        and int(parts[0]) == 1
        and int(parts[1]) >= 7
    )
    if not compatible:
        raise RuntimeError(
            "rextio-torch provider API 1.7 requires a compatible Rextio "
            "plugin host API in major 1 with minor >= 7; this environment "
            f"advertises PLUGIN_API_VERSION={PLUGIN_API_VERSION!r}"
        )


class RextioTorchPlugin:
    """Plugin API 1.7 provider for bounded CPU and CUDA build-only surfaces."""

    plugin_id = PLUGIN_ID
    api_version = REQUIRED_PLUGIN_API

    def to_rextio_plugin(self) -> RextioPlugin:
        """Return the v1 metadata Rextio core registers this plugin under."""
        _require_compatible_host_api()
        from rextio.plugins.models import RextioPlugin

        from rextio_torch.rules import COVERAGE

        return RextioPlugin(
            id=PLUGIN_ID,
            name=f"PyTorch to tch (rextio-torch {__version__})",
            source_language="python",
            target_language="rust",
            packages=COVERAGE.packages,
        )

    def covers(self) -> CoverageDecl:
        """Return the packages, modules, and symbols this plugin covers."""
        _require_compatible_host_api()
        from rextio_torch.rules import COVERAGE

        return COVERAGE

    def describe(self, config: RextioConfig) -> tuple[RuleRecord, ...]:
        """Return the rule records for the resolved project configuration."""
        _require_compatible_host_api()
        from rextio_torch.rules import torch_rule_records

        del config
        return torch_rule_records()

    def type_vocabulary(self) -> tuple[PluginType, ...]:
        """Return the CPU and bounded build-only CUDA annotation vocabulary."""
        _require_compatible_host_api()
        from rextio_torch.plugin_types import plugin_types

        return plugin_types()

    def claim(self, site: ClaimSite, config: RextioConfig) -> ClaimResult:
        """Decide, at analysis time, whether this plugin lowers the site."""
        _require_compatible_host_api()
        from rextio_torch.claim import claim as claim_site

        return claim_site(site, config)

    def lower(self, claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr:
        """Emit the Rust expression for a previously claimed site."""
        _require_compatible_host_api()
        from rextio_torch.lower import lower as lower_site

        return lower_site(claimed, ctx)

    def function_scope_guard(
        self,
        ctx: PluginFunctionScopeContext,
    ) -> PluginFunctionScopeGuard | None:
        """Install one no-grad scope only around boundary-free native execution."""
        _require_compatible_host_api()
        from rextio.plugins.api import (
            LOWERING_BACKEND_PYO3,
            PluginFunctionScopeGuard,
        )

        if (
            ctx.backend != LOWERING_BACKEND_PYO3
            or ctx.has_python_boundary_calls
            or not ctx.used_rule_ids
        ):
            return None

        return PluginFunctionScopeGuard(rust="tch::no_grad_guard()")

    def crate_dependencies(self) -> tuple[CrateDependency, ...]:
        """Return the exact tch pin and python-extension feature."""
        _require_compatible_host_api()
        from rextio.plugins.api import CrateDependency

        return (
            CrateDependency(
                name="tch",
                version="=0.24.0",
                features=("python-extension",),
            ),
        )


def plugin() -> RextioTorchPlugin:
    """Entry-point factory for the ``rextio.plugins`` group."""
    return RextioTorchPlugin()
