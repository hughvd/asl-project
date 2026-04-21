"""Quick bench of dataloading speed to calibrate training wall-time."""
import time
import sys
sys.path.insert(0, "scripts")
from baseline_models import load_splits, make_loaders

splits, classes, c2i = load_splits()
tr, va, te = make_loaders(splits, c2i, size=64, batch=128)
t0 = time.time()
for i, (x, y) in enumerate(tr):
    if i == 10:
        break
print(f"10 batches of 128 @64x64 took {time.time() - t0:.1f}s")
print(f"Train batches/epoch: {len(tr)}")
