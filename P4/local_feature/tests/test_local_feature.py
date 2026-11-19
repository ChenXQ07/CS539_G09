import ast,io
from pathlib import Path
import pytest
import torch
from torch import nn
from p4_local import LocalFeatureBranch

@pytest.mark.parametrize("shape",[(4,64,8,8),(4,64,16,16),(4,64,32,32),(2,64,13,19)])
def test_shape(shape):
    m=LocalFeatureBranch();x=torch.randn(shape);y=m(x)
    assert y.shape==x.shape and y.device==x.device and y.dtype==x.dtype
    assert torch.isfinite(y).all()

@pytest.mark.parametrize("value,error",[(None,TypeError),(torch.zeros(2,63,8,8),ValueError),(torch.zeros(64,8,8),ValueError),(torch.zeros(0,64,8,8),ValueError),(torch.zeros(2,64,8,8,dtype=torch.long),TypeError)])
def test_contract(value,error):
    with pytest.raises(error):LocalFeatureBranch()(value)

@pytest.mark.parametrize("value",[0,-1,True,2.5])
def test_channels(value):
    with pytest.raises(ValueError):LocalFeatureBranch(value)

def reference():
    source=Path(__file__).resolve().parents[1]/'reference'/'UNetFormer_RSSS.py'
    tree=ast.parse(source.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ConvBN')
    namespace={'nn':nn};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
    class Reference(nn.Module):
        def __init__(self):
            super().__init__();self.local1=namespace['ConvBN'](64,64,kernel_size=3);self.local2=namespace['ConvBN'](64,64,kernel_size=1)
        def forward(self,x):return self.local2(x)+self.local1(x)
    return Reference()

@pytest.mark.parametrize("training",[False,True])
def test_source_forward_and_gradient_equivalence(training):
    torch.manual_seed(539);m=LocalFeatureBranch();r=reference()
    r.load_state_dict(m.state_dict(),strict=True);m.train(training);r.train(training)
    x=torch.randn(2,64,13,19,requires_grad=True);z=x.detach().clone().requires_grad_()
    y=m(x);expected=r(z);torch.testing.assert_close(y,expected,rtol=0,atol=0)
    target=torch.randn_like(y)
    ((y-target)**2).mean().backward();((expected-target)**2).mean().backward()
    torch.testing.assert_close(x.grad,z.grad,rtol=0,atol=0)
    for (name,p),(other,q) in zip(m.named_parameters(),r.named_parameters()):
        assert name==other;torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)

def test_optimizer_updates_both_paths():
    torch.manual_seed(539);m=LocalFeatureBranch();opt=torch.optim.SGD(m.parameters(),lr=.01)
    before={n:p.detach().clone() for n,p in m.named_parameters()}
    x=torch.randn(2,64,8,8,requires_grad=True);target=torch.randn_like(x)
    ((m(x)-target)**2).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for p in m.parameters():assert p.grad is not None and torch.isfinite(p.grad).all()
    opt.step()
    for name in ['local1.0.weight','local2.0.weight']:
        assert not torch.equal(before[name],dict(m.named_parameters())[name])

def test_batchnorm_and_checkpoint():
    m=LocalFeatureBranch();norm=m.local1[1];before=norm.running_mean.clone()
    m.train();m(torch.randn(2,64,8,8)+2);assert not torch.equal(before,norm.running_mean)
    m.eval();before=norm.running_mean.clone();x=torch.randn(2,64,8,8);expected=m(x)
    assert torch.equal(before,norm.running_mean)
    buf=io.BytesIO();torch.save(m.state_dict(),buf);buf.seek(0)
    restored=LocalFeatureBranch().eval();restored.load_state_dict(torch.load(buf,weights_only=True),strict=True)
    torch.testing.assert_close(restored(x),expected,rtol=0,atol=0)

def test_three_instances_are_independent():
    branches=nn.ModuleList([LocalFeatureBranch() for _ in range(3)])
    assert len({m.local1[0].weight.data_ptr() for m in branches})==3
