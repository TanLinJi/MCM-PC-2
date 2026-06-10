**MCM-PC: Reliability-Aware Multi-Cache Matrix for Test-Time Point Cloud Adaptation**

```
Input point cloud
   ↓
Frozen 3D encoder + frozen text encoder
   ↓
Global feature + local part features + text prototypes
   ↓
Reliability estimation:
   entropy + compactness + margin + global-local consistency
   ↓
Multi-Cache Matrix update:
   global confident cache
   global compact cache
   global boundary cache
   local confident cache
   local compact cache
   local boundary cache
   semantic anchor
   ↓
Reliability-gated fusion:
   text logits
   global cache logits
   local cache logits
   negative/boundary suppression
   ↓
Final prediction
```

