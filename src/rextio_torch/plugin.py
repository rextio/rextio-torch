"""The rextio-torch plugin object and entry-point factory.

Implements plugin API 1.3: describe/covers, annotation vocabulary, claim/lower,
and the exact ``tch =0.24.0`` crate pin with ``python-extension``. This module
never imports torch; user-facing types are also import-free.

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
        PluginType,
        RuleRecord,
    )
    from rextio.plugins.models import RextioPlugin

PLUGIN_ID = "rextio-torch"
REQUIRED_PLUGIN_API = "1.3"

__all__ = ["PLUGIN_ID", "REQUIRED_PLUGIN_API", "RextioTorchPlugin", "plugin"]


class RextioTorchPlugin:
    """Plugin API 1.3 provider for the Alpha AOT float32 CPU inference surface."""

    plugin_id = PLUGIN_ID
    api_version = REQUIRED_PLUGIN_API

    def to_rextio_plugin(self) -> RextioPlugin:
        """Return the v1 metadata Rextio core registers this plugin under."""
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
        from rextio_torch.rules import COVERAGE

        return COVERAGE

    def describe(self, config: RextioConfig) -> tuple[RuleRecord, ...]:
        """Return the rule records for the resolved project configuration."""
        from rextio_torch.rules import torch_rule_records

        del config
        return torch_rule_records()

    def type_vocabulary(self) -> tuple[PluginType, ...]:
        """Return the annotation vocabulary this plugin adds to the analyzer."""
        from rextio_torch.plugin_types import plugin_types

        return plugin_types()

    def claim(self, site: ClaimSite, config: RextioConfig) -> ClaimResult:
        """Decide, at analysis time, whether this plugin lowers the site."""
        from rextio_torch.claim import claim as claim_site

        return claim_site(site, config)

    def lower(self, claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr:
        """Emit the Rust expression for a previously claimed site."""
        from rextio_torch.lower import lower as lower_site

        return lower_site(claimed, ctx)

    def crate_dependencies(self) -> tuple[CrateDependency, ...]:
        """Return the exact tch pin and python-extension feature."""
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
