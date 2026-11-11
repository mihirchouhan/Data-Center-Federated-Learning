"""Deterministic fabricated telemetry; never opens M100 data."""
import numpy as np
from .schema import Telemetry, frozen_array

def generate(spec, clients=3, *, faults=True):
    if spec.profile != "synthetic" or type(clients) is not int or not 1 <= clients <= 8:
        raise PermissionError("Small synthetic profile only")
    n = (spec.end-spec.start)//spec.width
    if n > 4096:
        raise ValueError("Synthetic generator is bounded to 4096 bins per node")
    result=[]
    for i in range(clients):
        rng=np.random.default_rng(np.random.SeedSequence([spec.seed,i]))
        times=(spec.start+np.arange(n)*spec.width).astype("datetime64[ns]")
        q=np.arange(n,dtype=np.float64)
        daily=2*np.pi*((times.view("i8")//1_000_000_000+600)%86400)/86400
        weekly=2*np.pi*(((times.view("i8")//1_000_000_000+600)+3*86400)%604800)/604800
        power=.45+.025*i+.05*np.sin(q/18)+rng.normal(0,.002,n)
        temp=22+i+.8*np.sin(q/24)+rng.normal(0,.05,n)
        v=np.column_stack((power,40+10*np.sin(q/13),np.full(n,10),np.full(n,45),
            np.full(n,5),.5+.1*np.sin(q/17),temp,15+3*np.sin(q/22),np.full(n,65),
            np.sin(daily),np.cos(daily),np.sin(weekly),np.cos(weekly)))
        mask=np.ones(v.shape,dtype=bool)
        if faults and n>150:
            v[65+i,1]=np.nan               # irregular feature gap
            v[110+i,0]=-1                  # invalid power
            v[120+i,6]=-300                # invalid temperature
            v[140+i,2]=101                 # invalid percentage
            mask[100+i,6]=False             # finite but unobserved is not a target
            v[30:35,7:9]=np.nan            # weather unavailable; explicit mask handles it
            keep=np.ones(n,dtype=bool);keep[80+i:82+i]=False
            times,v,mask=times[keep],v[keep],mask[keep]
        result.append(Telemetry(f"synthetic-node-{i}",frozen_array(times),frozen_array(v),frozen_array(mask)))
    return tuple(result)
