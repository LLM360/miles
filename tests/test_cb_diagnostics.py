import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from miles.utils.cb_diagnostics import capture_advantages, collect_diagnostics, response_rows, summarize_rows


def batch(ids, advantages):
    tensors = [torch.tensor(a, dtype=torch.float32) for a in advantages]
    data = dict(sample_indices=ids, rewards=[0.] * len(ids), advantages=tensors,
                values=[torch.full_like(a, .4) for a in tensors],
                loss_masks=[torch.ones_like(a) for a in tensors],
                returns=[torch.zeros_like(a) for a in tensors])
    capture_advantages(data, tensors)
    return data


def distributed_worker(rank, rendezvous, output):
    os.environ["MILES_CB_DIAGNOSTICS"] = "1"
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        # Sibling groups are split across ranks; wrong-response counts are unequal.
        data = batch([0], [[1.]]) if rank == 0 else batch([1, 2, 3], [[0.], [0.], [0.]])
        ps = NS(cp=NS(size=1), intra_dp_cp=NS(size=2, group=dist.group.WORLD, gloo_group=dist.group.WORLD))
        result = collect_diagnostics(NS(ppo_baseline="value", n_samples_per_prompt=2), data, ps)
        assert result["diag/cb/wrong/used_positive_frac"] == .25
        assert result["diag/cb/groups/all_wrong"] == 2
        assert result["diag/cb/all_wrong/responses"] == 4
        assert "_cb_diag_raw_advantages" not in data
        Path(output, str(rank)).write_text("ok")
    finally:
        dist.destroy_process_group()


class DiagnosticsTest(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"MILES_CB_DIAGNOSTICS": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_masks_response_weighting_and_no_mutation(self):
        data = batch([0, 1], [[1., float("nan")], [-1., -1., -1.]])
        data["loss_masks"][0][1] = 0
        data["values"][0][1] = float("nan")
        state = torch.random.get_rng_state().clone()
        out = summarize_rows(response_rows(data), 2)
        self.assertEqual(out["diag/cb/wrong/used_positive_frac"], .5)  # not token-weighted .25
        self.assertEqual(out["diag/cb/wrong/raw_mean"], 0.)
        self.assertEqual(out["diag/cb/wrong/raw_std"], 1.)
        self.assertEqual(out["diag/cb/nonfinite_responses"], 0)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertTrue(torch.isnan(data["advantages"][0][1]))
        self.assertEqual(data["advantages"][1].tolist(), [-1.] * 3)

    def test_raw_used_and_cb_var_bonus_are_distinct(self):
        data = batch([0, 1], [[.06], [.06]])
        data["_cb_diag_shift"] = [.09, .09]
        data["advantages"] = [torch.tensor([-.2]), torch.tensor([-.2])]
        out = summarize_rows(response_rows(data), 2)
        self.assertAlmostEqual(out["diag/cb/all_wrong/shift_mean"], .09)
        self.assertEqual(out["diag/cb/all_wrong/nonpositive_to_positive_frac"], 1.)
        self.assertEqual(out["diag/cb/all_wrong/raw_positive_frac"], 1.)
        self.assertEqual(out["diag/cb/all_wrong/used_positive_frac"], 0.)

    def test_min_reference_and_critic_range(self):
        data = batch([0, 1], [[.2, 0.], [.2, 0.]])
        data["values"] = [torch.tensor([-.2, .4])] * 2
        data["group_baseline"] = [0., 0.]
        data["group_baseline_valid"] = [True, True]
        out = summarize_rows(response_rows(data), 2)
        self.assertAlmostEqual(out["diag/cb/wrong/value_mse"], .1)
        self.assertEqual(out["diag/cb/wrong/value_outside_01_frac"], .5)
        self.assertEqual(out["diag/cb/wrong/nonpositive_to_positive_frac"], .5)
        self.assertEqual(out["diag/cb/correct/responses"], 0)
        self.assertNotIn("diag/cb/correct/value_mse", out)

    def test_incomplete_and_nonfinite_groups_not_classified(self):
        data = batch([0, 1, 2], [[1.], [float("nan")], [1.]])
        out = summarize_rows(response_rows(data), 2)
        self.assertEqual(out["diag/cb/nonfinite_responses"], 1)
        self.assertEqual(out["diag/cb/complete_groups"], 0)

    def test_disabled_is_noop_and_cp_is_explicitly_unsupported(self):
        with patch.dict(os.environ, {"MILES_CB_DIAGNOSTICS": "0"}):
            data = batch([0], [[1.]])
            self.assertNotIn("_cb_diag_raw_advantages", data)
            self.assertEqual(collect_diagnostics(None, data, None), {})
        data = batch([0], [[1.]])
        self.assertEqual(collect_diagnostics(None, data, NS(cp=NS(size=2))), {"diag/cb/supported": 0.})
        self.assertNotIn("_cb_diag_raw_advantages", data)

    def test_cli_flag_lengths_truncation_and_token_vs_sequence_mse(self):
        with patch.dict(os.environ, {"MILES_CB_DIAGNOSTICS": "0"}):
            data = batch([0, 1], [[1.], [1., 1., 1.]])
            data["values"] = [torch.tensor([0.]), torch.tensor([1., 1., 1.])]
            data["returns"] = [torch.tensor([0.]), torch.tensor([0., 0., 0.])]
            data["truncated"] = [0, 1]
            capture_advantages(data, data["advantages"], NS(cb_diagnostics=True))
            ps = NS(cp=NS(size=1), intra_dp_cp=NS(size=1))
            out = collect_diagnostics(NS(cb_diagnostics=True, ppo_baseline="value", n_samples_per_prompt=2), data, ps)
        self.assertEqual(out["diag/cb/all/critic_target_mse_sequence_mean"], .5)
        self.assertEqual(out["diag/cb/all/critic_target_mse_token_mean"], .75)
        self.assertEqual(out["diag/cb/all/response_length_mean"], 2.)
        self.assertEqual(out["diag/cb/all/truncation_rate"], .5)
        self.assertEqual(out["diag/cb/all/v0_mse"], .5)
        self.assertEqual(out["diag/cb/prompt/group_mean_mse"], .25)
        self.assertEqual(out["diag/cb/all/position_q1/tokens"], 2)
        self.assertEqual(out["diag/cb/all/position_q1/critic_target_mse"], .5)
        self.assertEqual(out["diag/cb/all/position_q4/tokens"], 0)
        self.assertNotIn("diag/cb/all/position_q4/critic_target_mse", out)

    def test_two_rank_global_denominators(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(distributed_worker, args=("file://"+tmp+"/rendezvous", tmp), nprocs=2, join=True)
            self.assertEqual([Path(tmp, str(i)).read_text() for i in range(2)], ["ok", "ok"])


if __name__ == "__main__":
    unittest.main()
