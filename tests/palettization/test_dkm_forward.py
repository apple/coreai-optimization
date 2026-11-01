# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""CPU tests for ``dkm_forward``: forward, knobs, and backward."""

import pytest
import torch

from coreai_opt.palettization.kmeans.dkm._forward import dkm_forward


def _inputs(P=2, N=64, K=4, D=1, seed=0, requires_grad=False):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(P, N, D, generator=g, requires_grad=requires_grad)
    C = torch.randn(P, K, D, generator=g, requires_grad=requires_grad)
    return W, C


class TestDKMForward:
    @pytest.mark.parametrize("D", [1, 2, 4])
    @pytest.mark.parametrize("kmeans_max_iter", [1, 3])
    def test_shapes_and_finite(self, D, kmeans_max_iter):
        W, C = _inputs(D=D)
        W_hat, C_new = dkm_forward(W, C, tau=1e-4, kmeans_max_iter=kmeans_max_iter)
        assert W_hat.shape == W.shape
        assert C_new.shape == C.shape
        assert torch.isfinite(W_hat).all()
        assert torch.isfinite(C_new).all()

    def test_reconstructs_exact_k_valued_input(self):
        # Weight takes exactly K distinct values; with centroids seeded at those
        # values and a sharp softmax, the soft assignment is effectively hard, so
        # the reconstruction reproduces the input.
        vals = torch.tensor([-2.0, -1.0, 1.0, 2.0])
        W = vals.repeat(16).reshape(1, 64, 1)
        C = vals.reshape(1, 4, 1)
        W_hat, _ = dkm_forward(W, C, tau=1e-4, kmeans_max_iter=1)
        torch.testing.assert_close(W_hat, W, atol=1e-3, rtol=0)

    def test_enforce_zero_reserves_exact_zero_code(self):
        W, C = _inputs()
        _, C_new = dkm_forward(W, C, tau=1e-4, kmeans_max_iter=3, enforce_zero=True)
        # Each partition keeps an exact-zero centroid (the nearest-origin one is pinned).
        assert torch.all(C_new.norm(dim=-1).amin(dim=1) == 0)

    def test_zero_threshold_changes_output(self):
        # zero_threshold floors the softmax weights; a large floor visibly changes
        # the centroid update, proving the knob is live (not hardcoded).
        W, C = _inputs()
        _, C_small = dkm_forward(W, C, tau=1e-4, kmeans_max_iter=3, zero_threshold=1e-7)
        _, C_large = dkm_forward(W, C, tau=1e-4, kmeans_max_iter=3, zero_threshold=1e-1)
        assert not torch.allclose(C_small, C_large)

    def test_quantize_fn_applied_each_step(self):
        # A quantize_fn that snaps centroids to zero forces C_new to zero.
        W, C = _inputs()
        _, C_new = dkm_forward(
            W, C, tau=1e-4, kmeans_max_iter=2, quantize_fn=lambda c: torch.zeros_like(c)
        )
        assert torch.equal(C_new, torch.zeros_like(C_new))


class TestDKMBackward:
    def test_grad_flows_to_inputs(self):
        W, C = _inputs(requires_grad=True)
        W_hat, C_new = dkm_forward(W, C, tau=1e-2, kmeans_max_iter=2)
        (W_hat.pow(2).sum() + C_new.sum()).backward()
        for g in (W.grad, C.grad):
            assert g is not None
            assert torch.isfinite(g).all()
        # Gradients are non-trivial (the soft assignment is genuinely differentiable).
        assert W.grad.abs().sum() > 0
        assert C.grad.abs().sum() > 0

    def test_backward_matches_finite_difference(self):
        # dkm_forward casts internally to fp32, so use a smooth tau and a loose
        # tolerance for the central-difference check.
        W = torch.randn(1, 6, 1, requires_grad=True)
        C = torch.randn(1, 3, 1)
        tau = 1.0

        def loss(w):
            W_hat, _ = dkm_forward(w, C, tau=tau, kmeans_max_iter=1)
            return W_hat.pow(2).sum()

        loss(W).backward()
        analytic = W.grad.clone()

        h = 1e-2
        with torch.no_grad():
            for idx in [(0, 0, 0), (0, 3, 0), (0, 5, 0)]:
                wp = W.detach().clone()
                wp[idx] += h
                wm = W.detach().clone()
                wm[idx] -= h
                numeric = (loss(wp) - loss(wm)) / (2 * h)
                assert abs(numeric.item() - analytic[idx].item()) < 1e-1

    def test_enforce_zero_backward_is_finite(self):
        W, C = _inputs(requires_grad=True)
        W_hat, _ = dkm_forward(W, C, tau=1e-2, kmeans_max_iter=2, enforce_zero=True)
        W_hat.pow(2).sum().backward()
        assert torch.isfinite(W.grad).all()
        assert torch.isfinite(C.grad).all()
