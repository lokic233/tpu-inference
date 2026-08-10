# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Regression test for the gemma-4 P/D disagg KV-transfer spec bug.
#
# gemma-4 interleaves sliding_attention (num_kv_heads=16, head_dim=256) and
# full_attention (num_global_key_value_heads=4, global_head_dim=512) layers, so
# per-layer KV caches have DIFFERENT shapes. The consumer transfer spec must be
# built PER LAYER; a single kv_caches[0]-derived spec replicated across all layers
# (the pre-fix behavior) pairs a mismatched source/destination block in the P->D
# DMA copy and crashes Mosaic with 'tpu.enqueue_dma op DMA source/target shape mismatch'.
#
# This test exercises the spec-construction logic in isolation (no live TPU): it
# builds the per-layer spec list the way TPUConnectorWorker._get_kv_spec does after
# the fix and asserts that (a) heterogeneous per-layer shapes are preserved and
# (b) the old uniform-replication approach would have produced wrong shapes for
# the non-zeroth layer type.

import unittest


def build_specs_per_layer(layer_shapes, num_blocks):
    """Mirror of the FIXED _get_kv_spec: one spec per layer from its live shape."""
    specs = []
    for shape in layer_shapes:
        s = list(shape)
        s[0] = num_blocks
        specs.append(tuple(s))
    return specs


def build_specs_uniform(layer_shapes, num_blocks):
    """Mirror of the BUGGY _get_kv_spec: kv_caches[0] spec replicated * num_layers."""
    s = list(layer_shapes[0])
    s[0] = num_blocks
    return [tuple(s)] * len(layer_shapes)


class HeterogeneousKvSpecTest(unittest.TestCase):

    def _gemma4_layer_shapes(self):
        # [num_blocks, block_size, num_kv_heads, kv(=2), head_dim]-style factoring.
        # Interleave a sliding layer (16 heads, 256) and a full layer (4 heads, 512).
        sliding = [1024, 16, 16, 2, 256]
        full = [1024, 16, 4, 2, 512]
        # gemma-4 pattern: several sliding then a full, repeated.
        return [sliding, sliding, sliding, sliding, sliding, full] * 2

    def test_fixed_spec_preserves_per_layer_shapes(self):
        shapes = self._gemma4_layer_shapes()
        specs = build_specs_per_layer(shapes, num_blocks=8)
        # every layer's spec must match that layer's own factoring (only dim0 changes)
        for shape, spec in zip(shapes, specs):
            self.assertEqual(spec[0], 8)
            self.assertEqual(list(spec[1:]), list(shape[1:]))
        # heterogeneity is actually present (sliding vs full differ)
        distinct = {tuple(s[1:]) for s in specs}
        self.assertEqual(len(distinct), 2, "expected 2 distinct layer shapes (sliding+full)")

    def test_uniform_spec_is_wrong_for_heterogeneous_layers(self):
        # This documents the pre-fix bug: uniform replication yields a spec whose
        # tail shape does NOT match the full_attention layers -> DMA shape mismatch.
        shapes = self._gemma4_layer_shapes()
        uniform = build_specs_uniform(shapes, num_blocks=8)
        mismatches = [i for i, (shp, spc) in enumerate(zip(shapes, uniform))
                      if list(spc[1:]) != list(shp[1:])]
        self.assertTrue(mismatches,
                        "uniform spec should mismatch at least the full_attention layers")
        # specifically the full layers (16 heads/256 spec applied to 4 heads/512 dest)
        self.assertIn(5, mismatches)

    def test_homogeneous_model_unaffected(self):
        # Qwen3-style: all layers identical -> uniform == per-layer (no regression).
        shapes = [[1024, 16, 8, 2, 128]] * 28
        per_layer = build_specs_per_layer(shapes, 8)
        uniform = build_specs_uniform(shapes, 8)
        self.assertEqual(per_layer, uniform)


if __name__ == "__main__":
    unittest.main()
