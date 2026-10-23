"""Exercise the real loss dispatcher with CPU surrogates for model forwards."""
import ast
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from miles.utils.actor_loss_reduction import attach_token_scales, actor_reducer, SCALE_KEY, validate_actor_reduction
from miles.utils.cb_training_profile import apply_cb_profile, generation_budget

MILES=Path(__file__).resolve().parents[1]

def load_function(name,path,namespace):
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),namespace)
    return namespace[name]


def args(**kw):
    fields=dict(actor_loss_agg_mode='token-mean',global_batch_size=4,loss_type='policy_loss',
        calculate_per_token_loss=False,qkv_format='thd',share_backbone_critic=True,
        recompute_loss_function=False,use_dynamic_global_batch_size=False,allgather_cp=False,vf_coef=.1)
    return NS(**(fields|kw))


def parallel(size=1):
    return NS(cp=NS(size=1),intra_dp=NS(size=size,group=dist.group.WORLD),intra_dp_cp=NS(size=size))


def seq_reducer(total,lengths,masks,*a,**kw):
    return lambda x:sum((v*m).sum()/m.sum().clamp_min(1) for v,m in zip(x.split(lengths),masks))


def dispatcher(ps):
    def pg(a,b,x,reduce):
        loss=reduce(x.square());return loss,{'pg_loss':loss.detach()}
    def vf(a,b,x,reduce):
        loss=reduce((x-.3).square());return loss,{'value_loss':loss.detach()}
    scope=dict(torch=torch,get_parallel_state=lambda:ps,_nan_dbg_warn_long_batch=lambda *a:None,
        get_sum_of_sample_mean=seq_reducer,policy_loss_function=pg,value_loss_function=vf,
        checkpoint=lambda fn,*xs:fn(*xs))
    return load_function('loss_function',MILES/'miles/backends/training_utils/loss.py',scope)


def check_gradient(rank,size,recompute=False,warmup=False):
    # Same global optimizer batch, unequal lengths AND unequal valid-token counts across ranks.
    features=[torch.tensor([1.,2.]),torch.tensor([3.]),torch.tensor([4.,5.,6.]),torch.tensor([7.,8.,9.,10.])]
    masks=[torch.tensor([1.,0.]),torch.ones(1),torch.tensor([1.,1.,0.]),torch.ones(4)]
    local_ids=list(range(rank*4//size,(rank+1)*4//size))
    a=args(recompute_loss_function=recompute,share_backbone_critic_only=warmup);ps=parallel(size)
    data={'loss_masks':[masks[i] for i in local_ids]}
    attach_token_scales(a,data,ps)
    actual=dispatcher(ps)
    for layout in [[local_ids],[[i] for i in local_ids]]:
        # First element is a list of indices in both layouts.
        theta=torch.tensor(.2,requires_grad=True);critic=torch.tensor(.7,requires_grad=True)
        for ids in layout:
            x=torch.cat([features[i] for i in ids]); ms=[masks[i] for i in ids]
            batch=dict(loss_masks=ms,response_lengths=[len(m) for m in ms],total_lengths=[len(m)+2 for m in ms],
                current_values=critic*x,**{SCALE_KEY:[data[SCALE_KEY][local_ids.index(i)] for i in ids]})
            # The production schedule divides returned loss by num_microbatches.
            # Stub only the CUDA allocator helper; evaluate the real dispatch/scaling code.
            import sys
            with patch.dict(sys.modules,{'miles.utils.memory_utils':NS(release_cache_for_large_backward=lambda x:None)}):
                value,normalizer,_=actual(a,batch,len(layout),theta*x,True)
            assert normalizer==1
            (value/len(layout)).backward()
        gradient=torch.stack([torch.zeros(()) if theta.grad is None else theta.grad,critic.grad])
        if size>1:
            dist.all_reduce(gradient);gradient/=size
        t=torch.tensor(.2,requires_grad=True);v=torch.tensor(.7,requires_grad=True)
        expected=sum(((t*x).square()*m).sum() for x,m in zip(features,masks))/sum(m.sum() for m in masks)
        if warmup: expected=expected*0
        expected+=.1*sum((((v*x-.3).square()*m).sum()/m.sum()) for x,m in zip(features,masks))/4
        ref=torch.stack(torch.autograd.grad(expected,(t,v)))
        torch.testing.assert_close(gradient,ref)


def worker(rank,path):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=2)
    try:check_gradient(rank,2)
    finally:dist.destroy_process_group()


class TokenMeanTests(unittest.TestCase):
    def test_dispatch_actor_and_critic_accumulation(self):check_gradient(0,1)
    def test_recomputed_loss_and_critic_warmup(self):
        check_gradient(0,1,recompute=True)
        check_gradient(0,1,warmup=True)
    def test_two_rank_global_denominator(self):
        with tempfile.TemporaryDirectory() as t:mp.spawn(worker,args=(t+'/ rendezvous',),nprocs=2,join=True)
    def test_separate_optimizer_steps_and_masked_sequences(self):
        data={'loss_masks':[torch.zeros(2),torch.ones(1),torch.ones(3),torch.ones(4)]}
        a=args(global_batch_size=2);attach_token_scales(a,data,parallel())
        self.assertEqual(data[SCALE_KEY],[2.,2.,2/7,2/7])
        with self.assertRaises(ValueError):actor_reducer(a,{'loss_masks':[torch.ones(1)],SCALE_KEY:[2.,2/7]},None)
        with self.assertRaises(ValueError):attach_token_scales(a,{'loss_masks':[torch.zeros(2)]*2},parallel())
    def test_native_mode_unchanged_and_invalid_options_rejected(self):
        f=lambda x:x
        self.assertIs(actor_reducer(args(actor_loss_agg_mode=None),{},f),f)
        for overrides in [dict(calculate_per_token_loss=True),dict(context_parallel_size=2),dict(use_tis=True),dict(loss_type='custom_loss')]:
            with self.assertRaises(ValueError):validate_actor_reduction(args(**overrides))
    def test_dynamic_context_and_explicit_profile(self):
        a=args(cb_token_mean_16k=True,custom_generate_function_path='miles360.dgcv.recipe_data.generate',
               gamma=1.,lambd=1.,seq_length=10240,max_tokens_per_gpu=49152)
        apply_cb_profile(a)
        self.assertEqual(a.sglang_context_length,16384)
        self.assertEqual(a.seq_length,16384)
        self.assertTrue(a.cb_diagnostics)
        self.assertEqual(generation_budget(a.rollout_max_context_len,512,a.rollout_max_response_len),15872)
        self.assertEqual(generation_budget(a.eval_max_context_len,2048,a.eval_max_response_len),14336)
        with self.assertRaises(ValueError):generation_budget(16384,16384,16384)
        old=args(seq_length=10240);apply_cb_profile(old);self.assertEqual(old.seq_length,10240)

if __name__=='__main__':unittest.main()
