"""Independent CPU synthetic architecture test; no model training path."""
import torch
from torch import nn

class MultiOutputGRU(nn.Module):
    def __init__(self, *, node_count, feature_dim=14, hidden_size=64, layers=2, embedding_size=8):
        super().__init__()
        if (feature_dim != 14 or any(type(v) is not int or v <= 0
                for v in (node_count,hidden_size,layers,embedding_size))):
            raise ValueError("Invalid dimensions")
        self.node_count=node_count
        self.embedding=nn.Embedding(node_count,embedding_size)
        self.recurrent=nn.GRU(feature_dim-1+embedding_size,hidden_size,
                              num_layers=layers,batch_first=True)
        self.head=nn.Linear(hidden_size,2)
        self.requires_grad_(False)  # Phase 1.3 architecture-only; training needs a new version.

    def forward(self,inputs):
        if inputs.device.type!="cpu" or inputs.dtype!=torch.float32:
            raise ValueError("CPU float32 synthetic tensors required")
        if inputs.ndim!=3 or inputs.shape[1:]!=(36,14) or len(inputs)==0 or not torch.isfinite(inputs).all().item():
            raise ValueError("Expected finite [N,36,14]")
        ids=inputs[:,:,-1]
        if ((ids < 0).any().item() or (ids>=self.node_count).any().item()
                or not torch.equal(ids,ids.round())
                or not torch.equal(ids,ids[:,:1].expand_as(ids))):
            raise ValueError("Invalid or mixed node identity")
        sequence,_=self.recurrent(torch.cat((inputs[:,:,:-1],self.embedding(ids.long())),dim=-1))
        result=self.head(sequence[:,-1,:])
        if result.shape!=(len(inputs),2) or not torch.isfinite(result).all().item():
            raise ValueError("Invalid two-output architecture result")
        return result

def synthetic_shape_check(x,spec,node_count):
    if spec.profile!="synthetic":
        raise PermissionError("No real model initialization is authorized")
    if not hasattr(x,"shape") or x.shape[1:]!=(36,14) or len(x)==0:
        raise ValueError("Nonempty synthetic batch required")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(spec.seed)
        model=MultiOutputGRU(node_count=node_count,hidden_size=spec.hidden_size,
            layers=spec.layers,embedding_size=spec.embedding_size).cpu().eval()
        with torch.inference_mode():
            out=model(torch.tensor(x.copy(),dtype=torch.float32,device="cpu"))
    return {"input_shape":list(x.shape),"output_shape":list(out.shape),"finite":bool(torch.isfinite(out).all()),
            "device":"cpu","synthetic_forward_only":True,"backpropagation":False,
            "trainable_parameters_enabled":any(p.requires_grad for p in model.parameters())}
