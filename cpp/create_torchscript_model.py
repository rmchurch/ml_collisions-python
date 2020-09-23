#!/usr/bin python

import torch
import sys
sys.path.append('/scratch/gpfs/rmc2/ml_collisions/alps/')
from models import ReSeg

model = ReSeg()
filepath = 'auglag_best.pth.tar'
out = torch.load(filepath)
model.load_state_dict(out['state_dict'])
model.eval()

#trace model with TorchScript
batch_size = 12 #TODO: Does this have to match the trained model?
example = torch.rand(batch_size,2,32,32,dtype=torch.float32)
traced_script_module = torch.jit.trace(model, example)

#save traced model, to be imported to C++
traced_script_module.save('traced_auglag_best.pt')
