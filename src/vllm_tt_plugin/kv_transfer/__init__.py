# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Prefill/decode disaggregation for TT models (PHASE2_DESIGN.md).

Intentionally empty: the API-server process imports
``vllm_tt_plugin.kv_transfer.tt_connector`` (``KVConnectorFactory.
supports_hma_config`` and ``KVConnectorLogging``), so nothing here may import
``ttnn`` or the model tree. ``worker.py`` imports ``ttnn`` lazily.
"""
