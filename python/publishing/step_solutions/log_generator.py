from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from python.publishing.step_solutions.locale_templates import (
    StepSolutionTemplatePaths,
    resolve_solution_template_paths,
)
from python.publishing.step_solutions.models import (
    StepSolutionLogGenerationResult,
    StepSolutionPackagePaths,
    StepSolutionPuzzleInstance,
)
from python.publishing.step_solutions.paths import (
    legacy_input_json_path,
    user_log_path,
)
from python.publishing.step_solutions.progress import (
    ProgressTimer,
    print_progress,
)
from python.publishing.step_solutions.puzzle_instance_adapter import (
    instance_to_legacy_payload,
)


DEFAULT_LEGACY_ROOT = Path("python/step-by-step_solutions")


def write_legacy_input_json(
    instance: StepSolutionPuzzleInstance,
    paths: StepSolutionPackagePaths,
    locale: str,
    template_paths: StepSolutionTemplatePaths,
) -> Path:
    """
    Write one normalized JSON payload for the legacy user-log generator.

    The legacy engine may not consume this exact JSON directly yet; this file
    is still valuable because it is the stable publishing-side contract.

    Later, if the legacy engine needs a small adapter, it should adapt from
    this JSON format rather than from book puzzle records directly.
    """

    output_path = legacy_input_json_path(
        paths=paths,
        local_puzzle_code=instance.internal_puzzle_code,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = instance_to_legacy_payload(instance)
    payload.update(
        {
            "locale": locale,
            "visual_template_path": str(template_paths.visual_template_path),
            "message_template_path": str(template_paths.message_template_path),
            "expected_user_log_path": str(
                user_log_path(paths, instance.internal_puzzle_code)
            ),
        }
    )

    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return output_path


def generate_user_log_for_instance(
    instance: StepSolutionPuzzleInstance,
    paths: StepSolutionPackagePaths,
    locale: str,
    template_paths: Optional[StepSolutionTemplatePaths] = None,
    legacy_root: Path = DEFAULT_LEGACY_ROOT,
    legacy_command: Optional[str] = None,
    force: bool = False,
    skip_existing: bool = False,
    dry_run: bool = False,
) -> StepSolutionLogGenerationResult:
    """
    Generate one localized Excel user log for one puzzle.

    Output filename:
        user_logs/L-1-1_user_logs.xlsx

    This function intentionally owns only the publishing-side orchestration.
    It prepares:
        - stable JSON input
        - expected output path
        - template paths

    Then it calls the legacy solution-log engine through one bridge.

    By default the bridge uses an environment/configurable command. This keeps
    the rest of the publishing platform independent from old script details.
    """

    template_paths = template_paths or resolve_solution_template_paths(locale)

    expected_log_path = user_log_path(
        paths=paths,
        local_puzzle_code=instance.internal_puzzle_code,
    )
    expected_log_path.parent.mkdir(parents=True, exist_ok=True)

    legacy_input_path = write_legacy_input_json(
        instance=instance,
        paths=paths,
        locale=locale,
        template_paths=template_paths,
    )

    base_result = StepSolutionLogGenerationResult(
        book_id=instance.book_id,
        locale=locale,
        internal_puzzle_code=instance.internal_puzzle_code,
        external_puzzle_code=instance.external_puzzle_code,
        commercial_book_code=instance.commercial_book_code,
        commercial_problem_id=instance.commercial_problem_id,
        user_log_path=expected_log_path,
        legacy_input_path=legacy_input_path,
        status="planned",
    )

    if expected_log_path.exists() and skip_existing and not force:
        return StepSolutionLogGenerationResult(
            **{
                **base_result.to_dict(),
                "user_log_path": expected_log_path,
                "legacy_input_path": legacy_input_path,
                "status": "skipped_existing",
                "warnings": ["User log already exists and skip_existing=True."],
            }
        )

    if dry_run:
        return StepSolutionLogGenerationResult(
            **{
                **base_result.to_dict(),
                "user_log_path": expected_log_path,
                "legacy_input_path": legacy_input_path,
                "status": "dry_run",
                "warnings": ["Dry run only; Excel user log was not generated."],
            }
        )

    if expected_log_path.exists() and force:
        expected_log_path.unlink()

    try:
        _call_legacy_user_log_generator(
            legacy_input_path=legacy_input_path,
            output_xlsx_path=expected_log_path,
            locale=locale,
            template_paths=template_paths,
            legacy_root=legacy_root,
            legacy_command=legacy_command,
        )
    except Exception as exc:
        return StepSolutionLogGenerationResult(
            **{
                **base_result.to_dict(),
                "user_log_path": expected_log_path,
                "legacy_input_path": legacy_input_path,
                "status": "failed",
                "errors": [str(exc)],
            }
        )

    if not expected_log_path.exists():
        return StepSolutionLogGenerationResult(
            **{
                **base_result.to_dict(),
                "user_log_path": expected_log_path,
                "legacy_input_path": legacy_input_path,
                "status": "failed",
                "errors": [
                    "Legacy generator completed without raising an error, "
                    f"but expected workbook was not created: {expected_log_path}"
                ],
            }
        )

    return StepSolutionLogGenerationResult(
        **{
            **base_result.to_dict(),
            "user_log_path": expected_log_path,
            "legacy_input_path": legacy_input_path,
            "status": "ok",
        }
    )


def generate_user_logs_for_instances(
    instances: Iterable[StepSolutionPuzzleInstance],
    paths: StepSolutionPackagePaths,
    locale: str,
    template_paths: Optional[StepSolutionTemplatePaths] = None,
    legacy_root: Path = DEFAULT_LEGACY_ROOT,
    legacy_command: Optional[str] = None,
    force: bool = False,
    skip_existing: bool = False,
    dry_run: bool = False,
) -> List[StepSolutionLogGenerationResult]:
    """
    Generate localized Excel user logs for many puzzle instances.

    Emits one compact progress line per puzzle so long production runs are
    observable from the terminal.
    """

    resolved_templates = template_paths or resolve_solution_template_paths(locale)
    instance_list = list(instances)
    total = len(instance_list)

    results: List[StepSolutionLogGenerationResult] = []

    for index, instance in enumerate(instance_list, start=1):
        timer = ProgressTimer.start()

        expected_log_path = user_log_path(
            paths=paths,
            local_puzzle_code=instance.internal_puzzle_code,
        )

        print_progress(
            f"LOG {index}/{total}",
            (
                f"{instance.book_id}/{locale} "
                f"{instance.internal_puzzle_code} -> "
                f"{expected_log_path.name} started"
            ),
        )

        result = generate_user_log_for_instance(
            instance=instance,
            paths=paths,
            locale=locale,
            template_paths=resolved_templates,
            legacy_root=legacy_root,
            legacy_command=legacy_command,
            force=force,
            skip_existing=skip_existing,
            dry_run=dry_run,
        )
        results.append(result)

        detail = ""
        if result.step_count:
            detail = f" | steps={result.step_count}"
        if result.errors:
            detail = f" | error={_short_progress_error(result.errors[0])}"

        print_progress(
            f"LOG {index}/{total}",
            (
                f"{instance.book_id}/{locale} "
                f"{instance.internal_puzzle_code} -> "
                f"{result.user_log_path.name} {result.status.upper()}"
                f"{detail} | elapsed={timer.elapsed_text()}"
            ),
        )

    return results

def _short_progress_error(message: str, max_length: int = 180) -> str:
    """
    Keep console progress errors readable.
    """

    text = " ".join(str(message or "").split())
    if len(text) <= max_length:
        return text

    return text[: max_length - 3] + "..."

def _call_legacy_user_log_generator(
    legacy_input_path: Path,
    output_xlsx_path: Path,
    locale: str,
    template_paths: StepSolutionTemplatePaths,
    legacy_root: Path,
    legacy_command: Optional[str],
) -> None:
    """
    Isolated bridge to the old step-by-step solution generator.

    Preferred usage:
        pass --legacy-command explicitly, or set STEP_SOLUTION_LEGACY_COMMAND.

    The command may contain placeholders:
        {input_json}
        {output_xlsx}
        {locale}
        {visual_template}
        {message_template}
        {legacy_root}

    Example:
        python python/step-by-step_solutions/tool_logs/main.py
          --input-json {input_json}
          --output {output_xlsx}
          --locale {locale}
          --template {visual_template}
          --messages {message_template}

    If no command is provided, the function attempts a conservative default:
        python -m python.step-by-step_solutions.tool_logs.main ...

    That default may need a small legacy patch if the old script does not
    currently expose those arguments.
    """

    command_template = (
        legacy_command
        or os.environ.get("STEP_SOLUTION_LEGACY_COMMAND")
        or _default_legacy_command()
    )

    repo_root = Path.cwd()
    legacy_root_abs = Path(legacy_root).resolve()

    replacements: Dict[str, str] = {
        "input_json": str(Path(legacy_input_path)),
        "output_xlsx": str(Path(output_xlsx_path)),
        "locale": locale,
        "visual_template": str(Path(template_paths.visual_template_path)),
        "message_template": str(Path(template_paths.message_template_path)),
        "legacy_root": str(legacy_root_abs),
    }

    command = command_template.format(**replacements)

    env = os.environ.copy()

    # The legacy tool imports sibling packages like:
    #
    #   from generator.algo_human import ...
    #
    # That works only if python/step-by-step_solutions is on PYTHONPATH.
    # Keep cwd at repo root so all publishing-relative output paths still work,
    # but make the old generator package visible to the child process.
    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_parts = [
        str(legacy_root_abs),
        str(repo_root),
    ]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)

    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    completed = subprocess.run(
        command,
        shell=True,
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
    )

    if completed.returncode != 0:
        message = [
            "Legacy user-log generator failed.",
            f"Command: {command}",
            f"Exit code: {completed.returncode}",
        ]
        if completed.stdout.strip():
            message.append("STDOUT:")
            message.append(completed.stdout.strip())
        if completed.stderr.strip():
            message.append("STDERR:")
            message.append(completed.stderr.strip())
        raise RuntimeError("\n".join(message))


def _default_legacy_command() -> str:
    """
    Conservative default command.

    The actual legacy root is injected later through the {legacy_root}
    placeholder. That lets the workflow support a custom --legacy-root while
    keeping the old generator isolated.
    """

    python_exe = shutil.which("python") or sys.executable or "python"

    return (
        f'"{python_exe}" '
        '"{legacy_root}/tool_logs/main.py" '
        '--input-json "{input_json}" '
        '--output "{output_xlsx}" '
        '--locale "{locale}" '
        '--template "{visual_template}" '
        '--messages "{message_template}"'
    )