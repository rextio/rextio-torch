"""rextio-torch: private incubator Rextio plugin for a proven PyTorch slice.

Implements Rextio plugin API 1.3 for a float32 CPU functional-linear → ReLU →
mean(dim=1, keepdim=False) vertical slice lowered via tch 0.24.0
(``python-extension``). The package root re-exports the plugin facade eagerly;
that facade defers core analyzer/config/plugin-host imports so generated
runtimes can still import ``rextio_torch.types`` under a minimal ``rextio``
package.
"""

from rextio_torch.__about__ import __version__
from rextio_torch.plugin import RextioTorchPlugin, plugin

__all__ = ["RextioTorchPlugin", "__version__", "plugin"]
