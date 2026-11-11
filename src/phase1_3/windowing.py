"""Node-local bounded synthetic windows; no dense real-data preparation."""
import numpy as np
from .schema import WindowBatch, MASKED, frozen_array
from .validation import sanitize, validate_batch
from .split import split_for_origin

def build_windows(records, mapping, spec):
    if spec.profile != "synthetic":
        raise PermissionError("Real-data window generation is not implemented or authorized")
    mapping = tuple(mapping)
    if not mapping or len(set(mapping)) != len(mapping) or any(not isinstance(s,str) or not s for s in mapping):
        raise ValueError("Unique nonempty node-string mapping required")
    records = tuple(records)
    if len({r.node for r in records}) != len(records):
        raise ValueError("One series per node; duplicate node record")
    if any(r.node not in mapping for r in records):
        raise ValueError("Unknown node: cannot extend training mapping with evaluation nodes")
    buffers = {s:[] for s in ("train","val","test")}
    rejected = {"split_boundary":0,"missing_timestamp":0,"input_invalid":0,"target_invalid":0}
    diagnostics = {}
    for record in sorted(records,key=lambda v:mapping.index(v.node)):
        record, audit = sanitize(record,spec)
        diagnostics[record.node] = audit
        ns = record.times.view("i8")
        positions = {int(t):i for i,t in enumerate(ns)}
        node = mapping.index(record.node)
        for origin in range(spec.start, spec.end, spec.stride*spec.width):
            split = split_for_origin(origin,spec)
            if split is None:
                rejected["split_boundary"] += 1
                continue
            hist = origin + np.arange(-36,0)*spec.width
            future = origin + np.arange(6)*spec.width
            needed = np.r_[hist,future]
            if any(int(t) not in positions for t in needed):
                rejected["missing_timestamp"] += 1
                continue
            h = np.array([positions[int(t)] for t in hist])
            f = np.array([positions[int(t)] for t in future])
            values = record.values[h].copy()
            values[:,list(MASKED)] = 0
            if not np.isfinite(values).all():
                rejected["input_invalid"] += 1
                continue
            # Both targets must be original observations in all six future bins.
            future_values = record.values[f][:,[0,6]]
            if not record.observed[f][:,[0,6]].all() or not np.isfinite(future_values).all():
                rejected["target_invalid"] += 1
                continue
            y = np.array([future_values[:,0].mean(dtype=np.float64)*1000,
                          future_values[:,1].mean(dtype=np.float64)],dtype=np.float64)
            x = np.column_stack((values,np.full(36,node)))
            buffers[split].append((x,y,node,origin,hist,future))
    result = {}
    for split, rows in buffers.items():
        def arr(i, shape, dtype):
            return frozen_array(np.array([row[i] for row in rows],dtype=dtype).reshape(shape))
        n = len(rows)
        batch = WindowBatch(arr(0,(n,36,14),np.float64),arr(1,(n,2),np.float64),
            arr(2,(n,),np.int64),arr(3,(n,),"datetime64[ns]"),
            arr(4,(n,36),"datetime64[ns]"),arr(5,(n,6),"datetime64[ns]"),
            split,mapping,spec.digest)
        validate_batch(batch,spec)
        result[split] = batch
    total = len(records)*len(range(spec.start,spec.end,spec.stride*spec.width))
    assert sum(len(b) for b in result.values()) + sum(rejected.values()) == total
    return result, {"candidate_windows":total,"accepted":{s:len(b) for s,b in result.items()},
                    "exclusive_rejections":rejected,"per_node_validation":diagnostics}
