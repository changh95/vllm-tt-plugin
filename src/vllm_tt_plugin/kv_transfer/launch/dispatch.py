# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""``vllm.general_plugins`` entry point ``tt_serve_launcher``: re-route a container's
fixed ``vllm serve ...`` line into a TT-specific launcher.

The tt-model container tool always runs ``vllm serve <weights> <flags>`` as PID 1 and
lets a serve profile control only the argv tail and the environment.  vLLM calls
``load_general_plugins()`` while it BUILDS the serve parser (``AsyncEngineArgs.
add_cli_args``, before any argument is parsed), so this hook runs in that PID-1 process
first.  When the profile's environment carries ``TT_SERVE_LAUNCHER=<name>`` and the
process was invoked as ``... serve ...``, it ``os.execv``s the launcher module with
everything after ``serve`` -- same PID (docker stop's SIGTERM still lands on it), same
interpreter, the variable removed from the environment so the launcher's own children
(front-ends, engine ranks; they load the same plugin) see this hook as a no-op.

Without the variable, or in any process whose argv[1] is not ``serve``, nothing happens.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping, MutableMapping

ENV_VAR = "TT_SERVE_LAUNCHER"

# launcher name -> module run as ``python -m <module> <argv after "serve">``
LAUNCHERS: dict[str, str] = {
    "pd_container": "vllm_tt_plugin.kv_transfer.launch.pd_container",
}


def plan_exec(argv: list[str], env: Mapping[str, str]) -> list[str] | None:
    """The argv to exec (``[python, -m, <module>, *argv[2:]]``) or ``None`` when this
    process is not a ``serve`` invocation under ``TT_SERVE_LAUNCHER``.

    An unknown launcher name raises: the profile asked for a launcher this plugin does
    not ship, and booting a plain ``vllm serve`` instead would silently serve the wrong
    thing."""
    name = env.get(ENV_VAR, "")
    if not name:
        return None
    if argv[1:2] != ["serve"]:
        return None
    module = LAUNCHERS.get(name)
    if module is None:
        raise RuntimeError(
            f"{ENV_VAR}={name!r} names no launcher; one of {sorted(LAUNCHERS)}"
        )
    return [sys.executable, "-m", module, *argv[2:]]


def maybe_exec(
    argv: list[str] | None = None,
    env: MutableMapping[str, str] | None = None,
    execv: Callable[[str, list[str]], None] = os.execv,
) -> list[str] | None:
    """Entry-point body.  Execs (never returns) when ``plan_exec`` yields a plan,
    otherwise returns ``None``.  ``argv``/``env``/``execv`` are injectable for tests;
    the returned plan is what a fake ``execv`` received."""
    argv = list(sys.argv if argv is None else argv)
    env = os.environ if env is None else env
    plan = plan_exec(argv, env)
    if plan is None:
        return None
    env.pop(ENV_VAR, None)  # children are plain vllm processes
    sys.stdout.flush()
    sys.stderr.flush()
    execv(plan[0], plan)
    return plan


__all__ = ["ENV_VAR", "LAUNCHERS", "maybe_exec", "plan_exec"]
