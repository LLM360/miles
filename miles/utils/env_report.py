import dataclasses
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EditablePackageInfo:
    name: str
    version: str
    location: str


@dataclass
class GitRepoInfo:
    package_name: str
    location: str
    commit: str
    dirty: bool
    diff_stat: str


@dataclass
class NodeEnvReport:
    role: str
    rank: int
    launcher_env_report: dict[str, Any] | None
    editable_packages: list[EditablePackageInfo]
    git_repos: list[GitRepoInfo]
    full_pip_list: list[dict[str, str]]


def collect_and_print_node_env_report(
    *,
    role: str,
    rank: int,
    env_report_json: str,
) -> NodeEnvReport:
    """Collect environment info for this node, print to stdout, return structured report.

    Called during actor init. Only performs collection when env_report_json is non-empty.

    Args:
        role: Actor role, e.g. "training" or "rollout"
        rank: Actor rank
        env_report_json: JSON string from launcher (may contain launch config info)
    """
    launcher_report = None
    if env_report_json:
        try:
            launcher_report = json.loads(env_report_json)
        except json.JSONDecodeError:
            logger.warning("Failed to parse env_report_json", exc_info=True)

    editable_packages = _collect_editable_packages()

    git_repos = []
    for pkg in editable_packages:
        git_info = _collect_git_info(package_name=pkg.name, location=pkg.location)
        if git_info:
            git_repos.append(git_info)

    full_pip_list = _collect_full_pip_list()

    report = NodeEnvReport(
        role=role,
        rank=rank,
        launcher_env_report=launcher_report,
        editable_packages=editable_packages,
        git_repos=git_repos,
        full_pip_list=full_pip_list,
    )

    _print_report(report)
    return report


def _collect_editable_packages() -> list[EditablePackageInfo]:
    try:
        result = subprocess.run(
            ["pip", "list", "--editable", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.warning("pip list --editable failed: %s", result.stderr)
            return []
        packages = json.loads(result.stdout)
        editable_list = []
        for pkg in packages:
            location = _get_package_location(pkg["name"])
            editable_list.append(EditablePackageInfo(
                name=pkg["name"],
                version=pkg["version"],
                location=location,
            ))
        return editable_list
    except Exception:
        logger.warning("Failed to collect editable packages", exc_info=True)
        return []


def _get_package_location(package_name: str) -> str:
    try:
        result = subprocess.run(
            ["pip", "show", package_name],
            capture_output=True,
            text=True,
            timeout=15,
        )
        for line in result.stdout.splitlines():
            if line.startswith("Editable project location:"):
                return line.split(":", 1)[1].strip()
            if line.startswith("Location:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        logger.warning("Failed to get location for %s", package_name, exc_info=True)
    return ""


def _collect_git_info(*, package_name: str, location: str) -> GitRepoInfo | None:
    if not location or not os.path.isdir(location):
        return None
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=location,
        )
        if commit_result.returncode != 0:
            return None
        commit = commit_result.stdout.strip()

        diff_result = subprocess.run(
            ["git", "diff", "--stat"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=location,
        )
        diff_stat = diff_result.stdout.strip()
        dirty = bool(diff_stat)

        return GitRepoInfo(
            package_name=package_name,
            location=location,
            commit=commit,
            dirty=dirty,
            diff_stat=diff_stat,
        )
    except Exception:
        logger.warning(
            "Failed to collect git info for %s at %s", package_name, location, exc_info=True
        )
        return None


def _collect_full_pip_list() -> list[dict[str, str]]:
    try:
        result = subprocess.run(
            ["pip", "list", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.warning("pip list failed: %s", result.stderr)
            return []
        return json.loads(result.stdout)
    except Exception:
        logger.warning("Failed to collect full pip list", exc_info=True)
        return []


def _print_report(report: NodeEnvReport) -> None:
    header = f"========== ENV REPORT (role={report.role}, rank={report.rank}) =========="
    print(header)

    if report.launcher_env_report:
        print("--- Launcher Config ---")
        print(json.dumps(report.launcher_env_report, indent=2, default=str))

    print("--- Editable Packages ---")
    for pkg in report.editable_packages:
        print(f"  {pkg.name} {pkg.version} {pkg.location}")

    for repo in report.git_repos:
        print(f"--- Git Status: {repo.package_name} ({repo.location}) ---")
        print(f"  commit: {repo.commit}")
        print(f"  dirty: {'yes' if repo.dirty else 'no'}")
        if repo.diff_stat:
            for line in repo.diff_stat.splitlines():
                print(f"  {line}")

    print("--- Full Package List ---")
    for pkg in report.full_pip_list:
        print(f"  {pkg.get('name', '?')} {pkg.get('version', '?')}")

    print("=" * len(header))
