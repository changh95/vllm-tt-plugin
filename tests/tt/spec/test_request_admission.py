# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Which request controls a speculating server admits, over HTTP.

``TTPlatform.validate_request`` decides this from the model's declared accept
modes and narrow decode. A launch whose model serves ``logits`` applies
logprobs, structured output, ``allowed_token_ids``, ``bad_words`` and
``min_tokens`` in its accept walk and admits them. A launch whose model serves
only ``argmax_ids`` admits them when it supports narrow decode
(``TT_SPEC_DRAFT_POLICY=solo``), decoding such a request without speculation,
and otherwise refuses each with HTTP 400 rather than answering without it.
No accept walk applies ``logit_bias`` or ``min_p``: a launch with narrow decode
admits them and decodes such a request without speculation, and any other
speculating launch refuses them, in vLLM itself before the plugin.
"""

from __future__ import annotations

import pytest

CONTROLS = {
    "logprobs": {"logprobs": 1},
    "logprobs-0": {"logprobs": 0},
    "structured_outputs": {"structured_outputs": {"regex": "(2|pression)+"}},
    "allowed_token_ids": {"allowed_token_ids": [17, 4099]},
    "bad_words": {"bad_words": ["2pression"]},
    "min_tokens": {"min_tokens": 2, "stop_token_ids": [17]},
}


@pytest.mark.parametrize("name", list(CONTROLS))
def test_a_control_is_admitted_exactly_when_logits_or_narrow_decode_serve_it(
    spec_server, spec_config, ascending_prompt, record, name
):
    result = spec_server.complete(
        ascending_prompt(16, start=3000), max_tokens=4, **CONTROLS[name]
    )
    record(status=result.status, body=result.body)
    # As in test_mixed_sampling.py: TT_SPEC_DRAFT_POLICY=always launches
    # without narrow decode, solo with it.
    if "logits" in spec_config.accept_modes or spec_config.draft_policy != "always":
        assert result.status == 200, result.body
        return
    assert result.status == 400, result.body
    assert "cannot serve" in str(result.body), result.body


ORDINARY_DECODE_CONTROLS = {
    "logit_bias": {"logit_bias": {"17": 1.0}},
    "min_p": {"temperature": 0.8, "min_p": 0.1, "seed": 7},
}


@pytest.mark.parametrize("name", list(ORDINARY_DECODE_CONTROLS))
def test_logit_bias_and_min_p_are_admitted_exactly_with_narrow_decode(
    spec_server, spec_config, ascending_prompt, record, name
):
    result = spec_server.complete(
        ascending_prompt(16, start=3100),
        max_tokens=4,
        **ORDINARY_DECODE_CONTROLS[name],
    )
    record(status=result.status, body=result.body)
    if spec_config.draft_policy != "always":
        assert result.status == 200, result.body
        return
    assert result.status == 400, result.body
