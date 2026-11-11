"""New strict split assignment: both input and target must be contained."""
def split_for_origin(origin, spec):
    for split in ("train","val","test"):
        lo,hi = spec.bounds(split)
        if origin-36*spec.width >= lo and origin+6*spec.width <= hi:
            return split
    return None
