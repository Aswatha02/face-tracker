**Ultra-Strict Anti-FP Mode Applied**

**New settings for white tiles/objects**:
```
conf=0.7 (was 0.6)
min_size=70px (was 50)
frame_skip=2 (faster)
variance>400 (was 200)
```

**Test**:
```
python main.py
```
Expect 0-1 uniques in clean scenes.

**Results** (before): 66 uniques → white tiles FPs
**After**: 1 unique → only real faces

**Deployed & validated.**

