"""Checked padding/truncation/masking tutorial, not a trained model benchmark."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import torch
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence

ROOT=Path(__file__).resolve().parent


def installed_version(name):
    """Record optional render dependencies without requiring them numerically."""
    try:return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:return None


def lengths_for(x,lengths):
    if not isinstance(x,torch.Tensor) or x.ndim!=3 or min(x.shape)<1:
        raise ValueError('Expected nonempty batch,time,feature tensor')
    if not isinstance(lengths,(tuple,list)) or len(lengths)!=x.shape[0]:
        raise ValueError('One explicit length per batch row required')
    if any(not isinstance(n,int) or isinstance(n,bool) or not 1<=n<=x.shape[1] for n in lengths):
        raise ValueError('Lengths must be positive integers no larger than padded time')
    return torch.tensor(lengths,dtype=torch.int64,device='cpu')


def worked_lstm():
    model=torch.nn.LSTM(1,1,batch_first=True,dtype=torch.float64)
    with torch.no_grad():
        model.weight_ih_l0.fill_(.5);model.weight_hh_l0.fill_(.5)
        model.bias_ih_l0.fill_(.1);model.bias_hh_l0.zero_()
    return model.eval()


def scalar_trace(values):
    h=c=0.;trace=[]
    for x in values:
        z=.5*float(x)+.5*h+.1
        gate=1/(1+math.exp(-z));candidate=math.tanh(z)
        c=gate*c+gate*candidate;h=gate*math.tanh(c)
        trace.append({'x':float(x),'z':z,'i_f_o':gate,'g':candidate,'c':c,'h':h})
    return trace


def packed(model,x,lengths):
    lens=lengths_for(x,lengths)
    return model(pack_padded_sequence(x,lens,batch_first=True,enforce_sorted=False))


def example():
    torch.set_num_threads(1)
    sequences=[torch.tensor([[1.],[2.]],dtype=torch.float64),torch.tensor([[0.],[1.],[-1.],[2.]],dtype=torch.float64)]
    x=pad_sequence(sequences,batch_first=True,padding_value=0.)
    lengths=[2,4];lens=lengths_for(x,lengths);mask=torch.arange(4)[None,:]<lens[:,None]
    model=worked_lstm();output,(plain_h,plain_c)=model(x)
    _,(packed_h,packed_c)=packed(model,x,lengths)
    gathered=output[torch.arange(2),lens-1]
    standalone=[];errors=[]
    for row,seq in enumerate(sequences):
        out,(h,c)=model(seq[None,:,:]);oracle=scalar_trace(seq[:,0].tolist())
        errors.extend([abs(float(out[0,t,0])-v['h']) for t,v in enumerate(oracle)])
        assert torch.allclose(packed_h[:,row],h[:,0],atol=1e-14,rtol=0)
        assert torch.allclose(packed_c[:,row],c[:,0],atol=1e-14,rtol=0)
        assert abs(float(h[0,0,0])-oracle[-1]['h'])<1e-14
        assert abs(float(c[0,0,0])-oracle[-1]['c'])<1e-14
        standalone.append({'h':float(h[0,0,0]),'c':float(c[0,0,0]),'scalar_trace':oracle})
    for row in range(2):
        oracle=scalar_trace(x[row,:,0].tolist())
        errors.extend([abs(float(output[row,t,0])-v['h']) for t,v in enumerate(oracle)])
    assert max(errors)<1e-14
    assert torch.allclose(gathered,packed_h[0],atol=1e-14,rtol=0)
    altered=x.clone();altered[~mask]=9.
    changed_output,(changed_h,_)=model(altered)
    _,(changed_packed_h,_)=packed(model,altered,lengths)
    assert torch.allclose(output[mask],changed_output[mask],atol=1e-14,rtol=0)
    assert torch.allclose(packed_h,changed_packed_h,atol=1e-14,rtol=0)
    assert not torch.allclose(plain_h[:,0],changed_h[:,0],atol=1e-8,rtol=0)
    masked_loss=output[:,:,0].square()[mask].sum()/mask.sum()
    naive_loss=output[:,:,0].square().mean()
    # Same forward tensor; changing reduction cannot erase its padded timesteps.
    assert output.shape==(2,4,1) and int(mask.sum())==6
    short=model(sequences[1][:2][None,:,:])[1][0]
    wrong=packed(model,x,[1,4])[1][0]
    differentiable=x.clone().requires_grad_(True)
    h=packed(model,differentiable,lengths)[1][0];h.sum().backward()
    assert torch.equal(differentiable.grad[~mask],torch.zeros_like(differentiable.grad[~mask]))
    source_names=['padding_demo.py','padding/2026-10-31/PROTOCOL.md']
    return {'date':'2026-10-31','kind':'authored synthetic teaching example; hand-set untrained weights',
        'model':{'input_size':1,'hidden_size':1,'batch_first':True,'layers':1,'directions':1,'dtype':'float64','device':'cpu','weights_ih_hh':.5,'bias_ih':.1,'bias_hh':0.,'initial_h_c':0.,'dropout':0.,'training_performed':False},
        'inputs':{'sequences':[[1.,2.],[0.,1.,-1.,2.]],'lengths':lengths,'right_padded':x.tolist(),'valid_mask':mask.tolist()},
        'ordinary_output_h':output.detach().tolist(),'ordinary_final_h':plain_h.detach().flatten().tolist(),
        'ordinary_final_c':plain_c.detach().flatten().tolist(),'last_valid_h':gathered.detach().flatten().tolist(),
        'packed_final_h':packed_h.detach().flatten().tolist(),'packed_final_c':packed_c.detach().flatten().tolist(),
        'standalone':standalone,'independent_scalar_max_abs_h_error':max(errors),
        'padding_changed_to9':{'ordinary_final_h':changed_h.detach().flatten().tolist(),'packed_final_h':changed_packed_h.detach().flatten().tolist(),'valid_outputs_unchanged':True},
        'loss_mask':{'target':0.,'loss':'mean squared output; denominator valid steps','valid_terms':6,'padded_terms':8,'masked':float(masked_loss.detach()),'naive_all_padded':float(naive_loss.detach()),'forward_states_unchanged_by_loss_mask':True},
        'truncation':{'B_first_two_final_h':float(short[0,0,0]),'A_incorrect_length1_final_h':float(wrong[0,0,0])},
        'packed_padded_input_gradients_zero':True,'limits':['Right-padded valid prefixes only; explicit correct lengths','Last-valid gathering equivalence here is for a forward-only LSTM, not bidirectional states','Loss masking does not skip recurrent computation','No real-data accuracy, speed or architecture superiority claim'],
        'environment':{'python':platform.python_version(),'torch':torch.__version__,'processor':platform.processor(),'torch_threads':torch.get_num_threads(),'dependencies':{n:installed_version(n) for n in ['numpy','matplotlib']}},
        'source_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in source_names}}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default=str(ROOT/'padding/2026-10-31/results.json'));a=p.parse_args()
    result=example();result['executed_at']=datetime.now(timezone.utc).isoformat();path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n');print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
