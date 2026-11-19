"""Synthetic connection example; projection adapters are NOT a full decoder."""
import sys,json
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'encoder_reference')]
from my_work.custom_encoder import ResNet18Encoder
from p4_local import LocalFeatureBranch

def main():
    torch.set_num_threads(2);torch.manual_seed(539)
    encoder=ResNet18Encoder(weights=None).train()
    # Smoke-test adapters. Use actual shared decoder preparation at integration.
    projections=nn.ModuleList([nn.Conv2d(c,64,1,bias=False) for c in (512,256,128)])
    norms=nn.ModuleList([nn.BatchNorm2d(64) for _ in range(3)])
    branches=nn.ModuleList([LocalFeatureBranch() for _ in range(3)])
    params=list(encoder.parameters())+list(projections.parameters())+list(norms.parameters())+list(branches.parameters())
    opt=torch.optim.SGD(params,lr=.01,momentum=.9,weight_decay=.0005)
    features=encoder(torch.randn(2,3,256,256))
    old=branches[0].local1[0].weight.detach().clone()
    outputs=[m(n(p(f))) for f,p,n,m in zip(reversed(features[1:]),projections,norms,branches)]
    loss=sum(y.square().mean() for y in outputs) # Synthetic objective, not segmentation loss.
    opt.zero_grad();loss.backward()
    assert encoder.conv1.weight.grad is not None and encoder.conv1.weight.grad.abs().sum()>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for m in branches for p in m.parameters())
    opt.step();assert not torch.equal(old,branches[0].local1[0].weight)
    report={'encoder_features':[list(f.shape) for f in features],
      'local_outputs_deep_to_shallow':[list(y.shape) for y in outputs],
      'encoder_gradients':True,'local_parameter_update':True,'initialization':'weights=None',
      'data':'synthetic','limitation':'test adapters; no global/shared decoder or segmentation accuracy'}
    (ROOT/'verification'/'encoder_connection.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
