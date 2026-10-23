"""Response views must preserve gradients without separate packed buffers."""
from argparse import Namespace

import pytest
import torch

from miles.backends.training_utils.loss import get_responses
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state


@pytest.mark.parametrize('layout', ['thd', 'bshd'])
@pytest.mark.parametrize('temperature', [1., .7])
@pytest.mark.parametrize('channels', [1, 17])
@pytest.mark.parametrize('partial', [False, True])
def test_response_views_and_masked_gradients(layout, temperature, channels, partial):
    single = GroupInfo(0, 1, None)
    set_parallel_state(ParallelState(single, single, single, single))
    args = Namespace(qkv_format=layout, rollout_temperature=temperature, allgather_cp=False)
    total, response = [5, 8, 6], [3, 5, 4]
    tokens = [torch.arange(n) for n in total]
    torch.manual_seed(7)
    shape = (1, 19, channels) if layout == 'thd' else (3, 8, channels)
    x = torch.randn(shape, requires_grad=True)
    actual = list(get_responses(x, args=args, unconcat_tokens=tokens,
        total_lengths=total, response_lengths=response, max_seq_lens=[8]*3))
    flat = x.reshape(-1, channels) / temperature
    spans = [(1, 4), (7, 12), (14, 18)] if layout == 'thd' else [(1, 4), (10, 15), (17, 21)]
    expected = [flat[start:end] for start, end in spans]
    selected = [0, 2] if partial else [0, 1, 2]
    weights = [torch.randn_like(v) for v in expected]
    for i, ((values, actions), reference) in enumerate(zip(actual, expected)):
        torch.testing.assert_close(values, reference, rtol=0, atol=0)
        torch.testing.assert_close(actions, tokens[i][-response[i]:])
        if temperature == 1.:
            assert values.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    # Reusing each output in a nonlinear loss tests branch accumulation;
    # omitting the middle output tests unused gradient handling and padding.
    got = sum((actual[i][0].square() * weights[i]).sum() + actual[i][0].sum() for i in selected)
    want = sum((expected[i].square() * weights[i]).sum() + expected[i].sum() for i in selected)
    actual_grad, = torch.autograd.grad(got, x, create_graph=True)
    expected_grad, = torch.autograd.grad(want, x, create_graph=True)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    actual_second, = torch.autograd.grad(actual_grad.sum(), x)
    expected_second, = torch.autograd.grad(expected_grad.sum(), x)
    torch.testing.assert_close(actual_second, expected_second, rtol=0, atol=0)
