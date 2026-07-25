"""rextio-torch: public Alpha Rextio plugin for a proven PyTorch slice.

Implements Rextio plugin API 1.6 for the proven float32 CPU surface and a
bounded, build-only CUDA E2 vertical-slice candidate lowered via tch 0.24.0
(``python-extension``). The package root re-exports the plugin facade eagerly;
that facade defers core analyzer/config/plugin-host imports so generated
runtimes can still import ``rextio_torch.types`` under a minimal ``rextio``
package.
"""

from rextio_torch.__about__ import __version__
from rextio_torch.plugin import RextioTorchPlugin, plugin

__all__ = ["RextioTorchPlugin", "__version__", "plugin"]
