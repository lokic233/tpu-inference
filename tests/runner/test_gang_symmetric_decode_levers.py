# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Gang-symmetric continue_decode combined with the continue_decode levers.

Under GANG_SYMMETRIC_DISPATCH the fused decode loop is bounded by
static_max_decode_steps and masks the iterations past the safe window: each
masked iteration re-runs the inputs of the last in-window step with every row
inactive (see test_gang_symmetric_decode.py). These tests check that the
masked iterations stay invisible when the opt-in loop levers are on:

- CONTINUE_DECODE_DEAD_KV_CLAMP shortens the attention seq_lens of rows that
  are inactive when a step starts. A masked iteration must build the same
  attention metadata as the step it repeats, or the clamp would move the KV
  write of a request that is only paused (not finished) to slot 0 of its
  first page and overwrite the KV of its first token.
- CONTINUE_DECODE_ALL_DONE_EXIT stops the loop once no row is active. Rows
  that are only paused by the window must not count as done, otherwise the
  trip count would depend on the window again. Rows that have really ended
  still stop the loop, so with this lever on the gang trip count depends on
  the data whenever the exit fires.
- CONTINUE_DECODE_SKIP_MIXED_RPA only changes how model_fn is traced.

The flags are module constants read at import, so the tests set them on the
module. The levers' runner config gates are not exercised here.
"""

from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.runner import test_gang_symmetric_decode as gsd
from tests.runner.test_gang_symmetric_decode import EOS, PAD, VOCAB, Req
from tpu_inference.runner import decode_loop
from tpu_inference.runner.decode_loop import continue_decode
from tpu_inference.runner.tpu_runner import TPUModelRunner


@pytest.fixture(autouse=True)
def _cpu_compatible_decode_core(monkeypatch):
    """XLA:CPU rejects the TPU-only compiler options the fused loop is built
    with; drop them off-TPU so the loop semantics can be checked anywhere."""
    if jax.default_backend() != "tpu":
        monkeypatch.setattr(decode_loop, "get_step_fn_compiler_options",
                            lambda: {})
        monkeypatch.setattr(decode_loop, "get_sampling_compiler_options",
                            lambda: {})
    decode_loop._get_decode_core.cache_clear()
    yield
    decode_loop._get_decode_core.cache_clear()


def _rpa_like_model(batch_size, dp_size, eos_pos):
    """The fake model of test_gang_symmetric_decode.py with RPA's KV placement.

    The new token's KV goes to slot seq_lens - 1 of the attention metadata and
    attention reads the slots before it, as in ragged paged attention. The
    dead-row KV clamp acts on those seq_lens, so this model sees what the
    clamp changes. kv_caches[1] holds the highest slot each row wrote.
    """
    eos_pos = jnp.asarray(eos_pos, dtype=jnp.int32)
    rows_per_dp = batch_size // dp_size

    def model_fn(state, kv_caches, input_ids, attn_metadata, *args, **kwargs):
        slots, max_slot, calls, folded = kv_caches
        pos = attn_metadata.input_positions
        seq_lens = attn_metadata.seq_lens.reshape(dp_size, -1)
        seq_lens = seq_lens[:, :rows_per_dp].ravel()
        real = seq_lens > 0
        rows = jnp.arange(batch_size)
        width = slots.shape[1]
        col = jnp.clip(seq_lens - 1, 0, width - 1)
        before = jnp.arange(width)[None, :] < col[:, None]
        context = jnp.sum(jnp.where(before, slots, 0), axis=1)
        next_tokens = (input_ids * 7 + pos * 13 + context) % (VOCAB - 2) + 2
        next_tokens = jnp.where(pos == eos_pos, EOS, next_tokens)
        slots = slots.at[rows,
                         col].set(jnp.where(real, input_ids, slots[rows, col]))
        max_slot = jnp.where(real, jnp.maximum(max_slot, col), max_slot)
        folded = folded + real.astype(jnp.int32)
        return ([slots, max_slot, calls + 1,
                 folded], next_tokens.astype(jnp.float32), None, None)

    def compute_logits_fn(state, hidden_states, lora_metadata):
        return jax.nn.one_hot(hidden_states.astype(jnp.int32), VOCAB)

    def sample_fn(rng, mesh, logits, sampling_metadata, **kwargs):
        return jnp.argmax(logits, axis=-1).astype(jnp.int32), logits

    # continue_decode unwraps the stock sampler via __wrapped__.
    sample_fn.__wrapped__ = sample_fn
    return model_fn, compute_logits_fn, sample_fn


def _set_levers(monkeypatch,
                *,
                dead_kv=False,
                all_done_every=0,
                skip_mixed=False):
    monkeypatch.setattr(decode_loop, "_DEADKV_ON", dead_kv)
    if dead_kv:
        # What _deadkv_config_gate records once a runner config passes.
        monkeypatch.setitem(decode_loop._DEADKV, "gate", {
            "prefix_caching": False,
            "dp": 1
        })
    monkeypatch.setattr(decode_loop, "_ALLDONE_EVERY", all_done_every)
    monkeypatch.setattr(decode_loop, "_SKIPMIXED_ON", skip_mixed)
    if skip_mixed:
        from tpu_inference.kernels.ragged_paged_attention.v3 import \
            kernel as rpa_kernel
        if rpa_kernel._skip_mixed_state is None:
            monkeypatch.setattr(rpa_kernel, "_skip_mixed_state",
                                rpa_kernel._skip_mixed_make_state())


def _run_rpa_window(monkeypatch, gang, reqs, **config):
    """test_gang_symmetric_decode._run_window with the RPA-like model."""
    with monkeypatch.context() as m:
        m.setattr(gsd, "_fake_model", _rpa_like_model)
        return gsd._run_window(monkeypatch, gang, reqs, **config)


def _prompt_slots(req):
    """The prompt KV test_gang_symmetric_decode._make_runner writes."""
    n = req.num_tokens
    return 2 + (np.arange(n - 1) * 5 + req.row) % (VOCAB - 2)


# A window limited to 12 steps by request 0 (500 of max_model_len 512 tokens).
# Request 1 is paused at the window end and continues in the next window;
# request 2 hits EOS inside the window.
WINDOW_REQS = [
    Req(num_tokens=500, row=0),
    Req(num_tokens=100, row=1),
    # Processes positions 199, 200, ...; emits EOS at position 203.
    Req(num_tokens=200, row=2, eos_pos=203),
]
WINDOW_CONFIG = dict(max_model_len=512, static_steps=32)


@pytest.mark.parametrize("eos_check_interval", [1, -1])
def test_gang_dead_kv_clamp_keeps_paused_rows_kv(monkeypatch,
                                                 eos_check_interval):
    """Dead-row KV clamp + masked iterations: a request that is only paused
    at the window end keeps its KV, and everything matches the stock loop
    with the same clamp."""
    _set_levers(monkeypatch, dead_kv=True)
    config = dict(WINDOW_CONFIG, eos_check_interval=eos_check_interval)
    stock = _run_rpa_window(monkeypatch, False, WINDOW_REQS, **config)
    gang = _run_rpa_window(monkeypatch, True, WINDOW_REQS, **config)

    if eos_check_interval > 0:
        # Stock early exit right after request 2's EOS step, in both paths.
        assert stock.calls == gang.calls == 5
    else:
        assert (stock.calls, gang.calls) == (12, 32)
        # The clamp did act in the stock loop: finished request 2 wrote its
        # padding KV to slot 0 of its first page.
        assert stock.slots[2, 0] != _prompt_slots(WINDOW_REQS[2])[0]
    steps = len(stock.tokens[0])
    for i in (0, 1):
        # Still running: prompt KV intact, only this window's slots written.
        r = WINDOW_REQS[i]
        np.testing.assert_array_equal(
            gang.slots[i, :r.num_tokens - 1],
            _prompt_slots(r),
            err_msg=f"request {i} lost prompt KV under the dead-row clamp")
        assert int(gang.max_pos[i]) == r.num_tokens - 1 + steps - 1
    assert int(gang.max_pos.max()) < 512
    gsd._assert_same_as_stock(gang, stock)


def _second_window(monkeypatch, gang, first):
    """Decodes request 1 of WINDOW_REQS for one more window, on the KV and
    with the last token the first window left it."""
    with monkeypatch.context() as m:
        m.setattr(gsd, "_fake_model", _rpa_like_model)
        if gang:
            m.setenv(gsd.GANG_ENV, "1")
        else:
            m.delenv(gsd.GANG_ENV, raising=False)
        req = Req(num_tokens=WINDOW_REQS[1].num_tokens + len(first.tokens[1]),
                  row=WINDOW_REQS[1].row)
        runner, sample_fn = gsd._make_runner([req],
                                             eos_check_interval=-1,
                                             **WINDOW_CONFIG)
        kv = list(runner.kv_caches)
        kv[0] = kv[0].at[req.row].set(jnp.asarray(first.slots[1]))
        runner.kv_caches = kv
        inputs = runner._prepare_inputs.return_value
        input_ids = inputs[0].at[req.row].set(first.tokens[1][-1])
        runner._prepare_inputs.return_value = (input_ids, ) + inputs[1:]
        with patch("tpu_inference.layers.jax.sample.sampling.sample",
                   sample_fn):
            TPUModelRunner._execute_continue_decode(runner, MagicMock())
    return list(runner.requests["req0"].output_token_ids)


def test_gang_dead_kv_clamp_next_window_tokens(monkeypatch):
    """The paused request reads its KV again in the next window, so its
    tokens there match the stock loop only if the masked iterations left
    that KV alone."""
    _set_levers(monkeypatch, dead_kv=True)
    tokens = {}
    for gang in (False, True):
        first = _run_rpa_window(monkeypatch,
                                gang,
                                WINDOW_REQS,
                                eos_check_interval=-1,
                                **WINDOW_CONFIG)
        tokens[gang] = (first.tokens[1],
                        _second_window(monkeypatch, gang, first))
    assert [len(t) for t in tokens[False]] == [12, 32]
    assert tokens[True] == tokens[False]


def _check_fixed_loop_matches_stock_loop(max_decode_steps):
    batch_size, width, static_steps = 4, 64, 9
    reqs = [(0, 20), (2, 40)]
    eos_pos = [-1, -1, 42, -1]  # row 2 emits EOS on its 4th step
    results = {}
    for fixed in (False, True):
        model_fn, compute_logits_fn, sample_fn = _rpa_like_model(
            batch_size, 1, eos_pos)
        kv, init_state = gsd._loop_inputs(batch_size, width, reqs)
        results[fixed] = continue_decode(
            state={},
            model_fn=model_fn,
            compute_logits_fn=compute_logits_fn,
            sample_fn=sample_fn,
            init_state=init_state,
            kv_caches=kv,
            max_decode_steps=jnp.array(max_decode_steps, jnp.int32),
            static_max_decode_steps=static_steps,
            eos_token_id=(EOS, ),
            padding_token_id=PAD,
            rng=jax.random.PRNGKey(0),
            mesh=None,
            sampling_metadata=None,
            continue_decode_eos_check_interval=-1,
            fixed_trip_count=fixed,
        )
    (tokens_s, kv_s, state_s, rng_s, _, _) = results[False]
    (tokens_f, kv_f, state_f, rng_f, _, _) = results[True]

    np.testing.assert_array_equal(tokens_f, tokens_s)
    assert int(state_f.step_counter) == int(state_s.step_counter)
    for name in ("current_tokens", "active_mask"):
        np.testing.assert_array_equal(getattr(state_f, name),
                                      getattr(state_s, name))
    for name in ("input_positions", "seq_lens"):
        np.testing.assert_array_equal(getattr(state_f.attn_metadata, name),
                                      getattr(state_s.attn_metadata, name))
    np.testing.assert_array_equal(np.asarray(rng_f), np.asarray(rng_s))
    np.testing.assert_array_equal(kv_f[0], kv_s[0])
    np.testing.assert_array_equal(kv_f[1], kv_s[1])
    # Row 0 never finishes, so its first slot still holds its prompt KV.
    assert int(np.asarray(kv_f[0])[0, 0]) == 2
    return kv_s, kv_f


@pytest.mark.parametrize("max_decode_steps", [1, 5, 9])
def test_fixed_trip_count_loop_with_dead_kv_clamp(monkeypatch,
                                                  max_decode_steps):
    """continue_decode(fixed_trip_count=True) with the clamp writes exactly
    the KV slots the non-fixed loop with the clamp writes."""
    _set_levers(monkeypatch, dead_kv=True)
    kv_s, kv_f = _check_fixed_loop_matches_stock_loop(max_decode_steps)
    assert int(kv_s[2]) == max_decode_steps
    assert int(kv_f[2]) == 9


@pytest.mark.parametrize("all_done_every", [1, 4])
@pytest.mark.parametrize("max_decode_steps", [1, 5, 9])
def test_fixed_trip_count_loop_with_all_loop_levers(monkeypatch,
                                                    max_decode_steps,
                                                    all_done_every):
    """Dead-row clamp, all-done exit and skip-MIXED tracing all on: the fixed
    trip count loop returns what the non-fixed loop with the same levers
    returns, and row 0 never ending keeps it at the static trip count."""
    _set_levers(monkeypatch,
                dead_kv=True,
                all_done_every=all_done_every,
                skip_mixed=True)
    kv_s, kv_f = _check_fixed_loop_matches_stock_loop(max_decode_steps)
    assert int(kv_s[2]) == max_decode_steps
    assert int(kv_f[2]) == 9


@pytest.mark.parametrize("all_done_every", [1, 4, 8])
def test_gang_all_done_exit_does_not_count_paused_rows(monkeypatch,
                                                       all_done_every):
    """All-done exit + masked iterations: rows paused by the window are not
    done, so the trip count stays static_max_decode_steps."""
    _set_levers(monkeypatch, all_done_every=all_done_every)
    reqs = [Req(num_tokens=500, row=0), Req(num_tokens=100, row=1)]
    config = dict(WINDOW_CONFIG, eos_check_interval=-1)
    stock = gsd._run_window(monkeypatch, False, reqs, **config)
    gang = gsd._run_window(monkeypatch, True, reqs, **config)
    assert stock.calls == 12
    assert gang.calls == 32, (
        f"gang-symmetric loop stopped after {gang.calls} iterations: the "
        "all-done exit counted rows paused by the window as done")
    gsd._assert_same_as_stock(gang, stock)
    assert int(gang.max_pos.max()) < 512


@pytest.mark.parametrize(
    "all_done_every, eos_pos, calls",
    [
        # Both requests end by step 5; the checkpoint after step 7 sees it.
        (4, (504, 202), (8, 8)),
        # They end at step 9, after the last in-window checkpoint (step 7):
        # the stock loop runs out its 12-step window and the gang loop stops
        # at the first checkpoint past it (step 15) instead of running 32.
        (8, (508, 202), (12, 16)),
    ])
def test_gang_all_done_exit_still_fires_when_rows_end(monkeypatch,
                                                      all_done_every, eos_pos,
                                                      calls):
    """When every request really ends inside the window, the all-done exit
    still stops the gang-symmetric loop, with the stock outputs."""
    _set_levers(monkeypatch, all_done_every=all_done_every)
    reqs = [
        Req(num_tokens=500, row=0, eos_pos=eos_pos[0]),
        Req(num_tokens=200, row=1, eos_pos=eos_pos[1]),
    ]
    config = dict(WINDOW_CONFIG, eos_check_interval=-1)
    stock = gsd._run_window(monkeypatch, False, reqs, **config)
    gang = gsd._run_window(monkeypatch, True, reqs, **config)
    assert (stock.calls, gang.calls) == calls
    gsd._assert_same_as_stock(gang, stock)
    assert all(t[-1] == EOS for t in gang.tokens)


def test_gang_levers_off_rpa_like_model(monkeypatch):
    """Levers off, RPA-like KV placement: the gang-symmetric loop still
    matches the stock loop (the RPA-like model itself adds nothing)."""
    _set_levers(monkeypatch)
    config = dict(WINDOW_CONFIG, eos_check_interval=-1)
    stock = _run_rpa_window(monkeypatch, False, WINDOW_REQS, **config)
    gang = _run_rpa_window(monkeypatch, True, WINDOW_REQS, **config)
    assert (stock.calls, gang.calls) == (12, 32)
    gsd._assert_same_as_stock(gang, stock)
