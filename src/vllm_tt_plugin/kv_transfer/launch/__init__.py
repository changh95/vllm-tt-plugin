# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Engine-process launch helpers for the Phase 3 PD pair (PHASE3_NOTES section 2).

* ``pd_launch_config`` -- pure Python (no vllm, no ttnn): the per-node ``vllm serve``
  argv, the ``kv_transfer_config`` JSON, ``pd_node_args.json`` and the tt-run rank
  binding YAML, all derived from one ``PairSettings`` so the two API front-ends and
  the two headless engine ranks build identical ``VllmConfig`` objects.
* ``pd_fabric_rank`` -- the program tt-run launches on every MPI rank.  It imports
  vllm and ttnn (lazily, inside ``main``) and runs a headless ``EngineCoreProc``
  in-process: rank 0 = prefill node (mesh 0), rank 1 = decode node (mesh 1).
* ``dispatch`` / ``pd_container`` -- the container entry point: the
  ``vllm.general_plugins`` hook ``tt_serve_launcher`` execs ``pd_container`` when
  ``TT_SERVE_LAUNCHER=pd_container``, and the supervisor turns the tool's one
  ``vllm serve`` line into front-ends + tt-run ranks + the in-process proxy
  (``data/`` holds the packaged two-mesh MGD and rank-binding template).

Nothing here is imported by the API server; the connector module stays ttnn-free.
"""
