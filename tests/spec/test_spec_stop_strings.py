# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stop strings when one step commits several tokens (speculative decoding).

``platform._install_tt_spec_stop_string_patch`` replays a multi-token update of
the frontend detokenizer one token at a time. The property under test: for any
split of a token stream into updates, the stop string, the output text and the
committed token ids are those of feeding the stream one token per update, i.e.
ordinary decoding.
"""

import random
from types import SimpleNamespace

import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import detokenizer as detokenizer_module

import vllm_tt_plugin  # noqa: F401
from vllm_tt_plugin.platform import _install_tt_spec_stop_string_patch

_install_tt_spec_stop_string_patch()

# A toy vocabulary: id -> text piece.
PIECES = [" An", " octopus", " has", " three", " hearts", ".", "\n", " it", "!", "..", "x", " end", ".\n", "\n."]


class ToyDetokenizer(detokenizer_module.BaseIncrementalDetokenizer):
    def __init__(self, params: SamplingParams):
        super().__init__(SimpleNamespace(sampling_params=params))

    def decode_next(self, next_token_id: int) -> str:
        return PIECES[next_token_id]


def _feed(params, chunks, eos_last=False):
    det = ToyDetokenizer(params)
    committed = []
    stop = None
    for i, chunk in enumerate(chunks):
        chunk = list(chunk)
        stop = det.update(chunk, eos_last and i == len(chunks) - 1)
        committed.extend(chunk)  # the caller passes the (trimmed) list on
        if stop is not None:
            break
    return stop, det.output_text, committed, list(det.output_token_ids)


def test_the_patch_is_installed_once():
    first = detokenizer_module.BaseIncrementalDetokenizer.update
    _install_tt_spec_stop_string_patch()
    assert detokenizer_module.BaseIncrementalDetokenizer.update is first
    assert getattr(first, "_tt_spec_stop_string_patch", False)


def test_the_earliest_stop_wins_not_the_first_listed():
    """The verifier's case: ' hearts' + '.' + '\\n' committed in one step."""
    params = SamplingParams(stop=["\n", "."])
    tokens = [0, 1, 2, 3, 4, 5, 6]
    plain = _feed(params, [[t] for t in tokens])
    spec = _feed(params, [[0], [1, 2], [3], [4, 5, 6]])
    assert plain[0] == "."
    assert plain[1] == " An octopus has three hearts"
    assert spec == plain


def test_token_ids_end_at_the_stopping_token():
    params = SamplingParams(stop=["."])
    spec = _feed(params, [[0, 5, 7]])
    assert spec[2] == [0, 5]
    assert spec[3] == [0, 5]


@pytest.mark.parametrize("include", [False, True])
@pytest.mark.parametrize("seed", range(200))
def test_any_split_equals_one_token_per_update(seed, include):
    rng = random.Random(seed)
    stops = rng.sample([".", "\n", "..", " end", "x!", "s."], k=rng.randint(1, 3))
    params = SamplingParams(
        stop=stops, include_stop_str_in_output=include, min_tokens=rng.choice([0, 0, 2])
    )
    tokens = [rng.randrange(len(PIECES)) for _ in range(rng.randint(1, 14))]
    eos_last = rng.random() < 0.3
    chunks, i = [], 0
    while i < len(tokens):
        n = rng.randint(1, 3)
        chunks.append(tokens[i : i + n])
        i += n
    plain_chunks = [[t] for t in tokens]
    assert _feed(params, chunks, eos_last) == _feed(params, plain_chunks, eos_last)


def test_no_stop_strings_takes_the_original_path():
    det = ToyDetokenizer(SamplingParams())
    ids = [0, 5, 6]
    assert det.update(ids, False) is None
    assert ids == [0, 5, 6]
    assert det.output_text == " An.\n"
