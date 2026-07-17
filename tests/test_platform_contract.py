"""Portable host contract: pins and no OS-only Linux rejection.

Certified real-Cargo evidence remains macOS arm64. Linux x86_64/aarch64 are
experimental smoke hosts; Windows is deferred. These tests guard the contract
without claiming Linux certification.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from rextio_torch.plugin import REQUIRED_PLUGIN_API, plugin

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src" / "rextio_torch"
E2E_ROOT = ROOT / "tests" / "e2e"

# Host probes that indicate OS/arch gating when used in code (not comments).
_PLATFORM_PROBE_ATTRS = frozenset(
    {
        "system",
        "machine",
        "platform",
        "uname",
        "architecture",
        "processor",
        "node",
        "release",
        "version",
    }
)
_OS_NAME_LITERALS = frozenset(
    {
        "darwin",
        "linux",
        "windows",
        "win32",
        "win64",
        "cygwin",
        "msys",
        "macos",
        "mac",
        "nt",
        "posix",
        "java",
        "aix",
        "freebsd",
        "openbsd",
        "netbsd",
        "sunos",
    }
)


def _python_sources(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if p.is_file())


def _call_name(node: ast.AST) -> str | None:
    """Return a dotted call target like ``platform.system`` or ``pytest.mark.skipif``."""
    if not isinstance(node, ast.Call):
        return None
    parts: list[str] = []
    cur: ast.AST = node.func
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def _is_sys_platform(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "platform"
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
    )


def _is_os_name(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "name"
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    )


def _is_platform_probe_call(node: ast.AST) -> bool:
    """``platform.system()`` / ``platform.machine()`` / etc."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr in _PLATFORM_PROBE_ATTRS
        and isinstance(func.value, ast.Name)
        and func.value.id == "platform"
    )


def _is_os_uname_call(node: ast.AST) -> bool:
    """``os.uname()`` is a Unix host probe sometimes used for OS gates."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "uname"
        and isinstance(func.value, ast.Name)
        and func.value.id == "os"
    )


def _const_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _compare_operands(node: ast.Compare) -> list[ast.AST]:
    return [node.left, *node.comparators]


def _os_gate_findings(tree: ast.AST, text: str, rel: str) -> list[str]:
    """Structural OS/arch gate evidence (AST only — ignores comments/docstrings)."""
    findings: list[str] = []

    def locate(node: ast.AST) -> str:
        line = getattr(node, "lineno", 0)
        snippet = (ast.get_source_segment(text, node) or type(node).__name__).strip()
        if len(snippet) > 80:
            snippet = snippet[:77] + "..."
        return f"{rel}:{line}: {snippet}"

    for node in ast.walk(tree):
        if _is_sys_platform(node):
            findings.append(locate(node))
            continue
        if _is_os_name(node):
            findings.append(locate(node))
            continue
        if _is_platform_probe_call(node) or _is_os_uname_call(node):
            findings.append(locate(node))
            continue

        # ``sys.getwindowsversion`` / similar host-specific sys helpers
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "sys"
            and node.attr in {"getwindowsversion", "winver"}
        ):
            findings.append(locate(node))
            continue

        # Comparison that mixes a host probe with an OS-name literal
        if isinstance(node, ast.Compare):
            operands = _compare_operands(node)
            has_probe = any(
                _is_sys_platform(op)
                or _is_os_name(op)
                or _is_platform_probe_call(op)
                or _is_os_uname_call(op)
                for op in operands
            )
            has_os_literal = any(
                (s := _const_str(op)) is not None and s.lower() in _OS_NAME_LITERALS
                for op in operands
            )
            if has_probe and has_os_literal:
                findings.append(locate(node))

    # Deduplicate while preserving order (Attribute + enclosing Compare may both hit).
    seen: set[str] = set()
    unique: list[str] = []
    for item in findings:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _skipif_has_os_gate(call: ast.Call) -> bool:
    """True when a pytest.mark.skipif condition structurally probes host OS."""
    for arg in (*call.args, *(kw.value for kw in call.keywords)):
        for child in ast.walk(arg):
            if _is_sys_platform(child) or _is_os_name(child):
                return True
            if _is_platform_probe_call(child) or _is_os_uname_call(child):
                return True
            if (
                isinstance(child, ast.Attribute)
                and isinstance(child.value, ast.Name)
                and child.value.id == "sys"
                and child.attr in {"getwindowsversion", "winver"}
            ):
                return True
            if isinstance(child, ast.Compare):
                for op in _compare_operands(child):
                    lit = _const_str(op)
                    if lit is not None and lit.lower() in _OS_NAME_LITERALS:
                        # Only count if the comparison also has a probe-like operand
                        # or a Name that could be platform result — require probe.
                        if any(
                            _is_sys_platform(x)
                            or _is_os_name(x)
                            or _is_platform_probe_call(x)
                            or _is_os_uname_call(x)
                            for x in _compare_operands(child)
                        ):
                            return True
    return False


def _skipif_mentions_cargo(call: ast.Call, text: str) -> bool:
    snippet = ast.get_source_segment(text, call) or ""
    return "cargo" in snippet.lower()


def test_plugin_source_has_no_os_only_host_rejection() -> None:
    """Plugin must not gate hosts via structural OS/arch probes (AST evidence).

    Harmless comments, docstrings, and documentation words are ignored because
    this walk uses the AST only — not raw text word scans.
    """
    offenders: list[str] = []
    for path in _python_sources(SRC_ROOT):
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        rel = str(path.relative_to(ROOT))
        offenders.extend(_os_gate_findings(tree, text, rel))
    assert offenders == [], (
        f"plugin source must not gate hosts by OS/arch probes; found: {offenders}"
    )


def test_crate_and_api_pins_remain_exact() -> None:
    """Native AOT pin contract is host-agnostic and must stay exact."""
    assert REQUIRED_PLUGIN_API == "1.3"
    deps = plugin().crate_dependencies()
    assert len(deps) == 1
    dep = deps[0]
    assert dep.name == "tch"
    assert dep.version == "=0.24.0"
    assert dep.features == ("python-extension",)


def test_e2e_real_cargo_skips_only_on_missing_cargo_not_os() -> None:
    """Real-Cargo modules may skip without cargo; they must not skipif on host OS."""
    for path in sorted(E2E_ROOT.glob("test_*.py")):
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        # Module body must not introduce OS probes used for host gating.
        module_gates = _os_gate_findings(tree, text, str(path.relative_to(ROOT)))
        assert module_gates == [], f"{path.name} contains OS probes: {module_gates}"

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            if name is None or not name.endswith("skipif"):
                continue
            assert not _skipif_has_os_gate(node), (
                f"{path.name} skipif gates on host OS: {ast.get_source_segment(text, node)}"
            )
            assert _skipif_mentions_cargo(node, text), (
                f"{path.name} skipif should be cargo-related, got: "
                f"{ast.get_source_segment(text, node)}"
            )


def test_e2e_env_setup_forbids_version_bypass_and_requires_use_pytorch() -> None:
    """E2E builders must set LIBTORCH_USE_PYTORCH=1 and strip the bypass."""
    for path in sorted(E2E_ROOT.glob("test_*.py")):
        text = path.read_text(encoding="utf-8")
        assert 'LIBTORCH_USE_PYTORCH"] = "1"' in text or "LIBTORCH_USE_PYTORCH'] = '1'" in text
        assert 'pop("LIBTORCH_BYPASS_VERSION_CHECK"' in text
        assert 'assert "LIBTORCH_BYPASS_VERSION_CHECK" not in os.environ' in text or (
            "assert 'LIBTORCH_BYPASS_VERSION_CHECK' not in os.environ" in text
        )
        # Must never assign the bypass to a truthy value.
        assert re.search(r'LIBTORCH_BYPASS_VERSION_CHECK"\s*=\s*"[^"]+"', text) is None


def test_linux_smoke_script_documents_pins_and_forbids_bypass() -> None:
    script = ROOT / "scripts" / "linux-smoke.sh"
    assert script.is_file(), "experimental Linux smoke script is missing"
    text = script.read_text(encoding="utf-8")
    assert "LIBTORCH_USE_PYTORCH=1" in text
    assert "LIBTORCH_BYPASS_VERSION_CHECK" in text
    assert "2.11.0" in text
    assert "needs_cargo" in text
    # Must unset / refuse bypass, not export it as an accepted path.
    assert "unset LIBTORCH_BYPASS_VERSION_CHECK" in text
    assert "export LIBTORCH_BYPASS_VERSION_CHECK" not in text
    # Fail closed on unlisted hosts (only Linux experimental + Darwin pin-check).
    assert 'system == "Linux"' in text
    assert 'system == "Darwin"' in text
    assert "SystemExit" in text
    assert "unsupported host system" in text or "deferred/unverified" in text
