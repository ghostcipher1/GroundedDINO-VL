# Migration from groundingdino to groundeddino_vl

This guide helps you migrate from the legacy `groundingdino` namespace to the modern `groundeddino_vl` package.

## Why Migrate?

The `groundingdino` namespace is now a **lightweight compatibility shim** that re-exports from `groundeddino_vl`. While the high-level API continues to work, internal module imports are no longer supported. Migrating to `groundeddino_vl` gives you:

- Access to the latest features and improvements
- Full access to all internal modules and utilities
- Cleaner, more maintainable code
- No deprecation warnings

## Quick Migration Table

| Old (groundingdino) | New (groundeddino_vl) | Status |
|---------------------|----------------------|--------|
| `from groundingdino import load_model` | `from groundeddino_vl import load_model` | ✓ Both work |
| `from groundingdino import predict` | `from groundeddino_vl import predict` | ✓ Both work |
| `from groundingdino import models` | `from groundeddino_vl import models` | ✓ Both work |
| `from groundingdino import util` | `from groundeddino_vl import utils` | ✓ Both work (note: util → utils) |
| `from groundingdino import datasets` | `from groundeddino_vl import data` | ✓ Both work (note: datasets → data) |
| `from groundingdino.util.misc import NestedTensor` | `from groundeddino_vl.utils.misc import NestedTensor` | ✗ Old no longer works |
| `from groundingdino.datasets.transforms import Compose` | `from groundeddino_vl.data.transforms import Compose` | ✗ Old no longer works |
| `from groundingdino.models import build_model` | `from groundeddino_vl.models import build_model` | ✗ Old no longer works |
| `import groundingdino._C` | `import groundeddino_vl._C` | ✗ Old no longer works |

## What Still Works

The **high-level public API** and **module-level access** continue to work via the compatibility shim (with deprecation warnings):

```python
# ✓ Still works - high-level API
from groundingdino import load_model, predict, load_image, annotate

# ✓ Still works - module-level access
from groundingdino import models, util, datasets

# Use modules like this:
import groundingdino
model = groundingdino.models.build_model(config)
```

## What No Longer Works

**Deep imports** into internal modules are no longer supported:

```python
# ✗ No longer works
from groundingdino.util.misc import NestedTensor
from groundingdino.datasets.transforms import Compose
from groundingdino.models.GroundingDINO import GroundingDINO

# ✓ Use groundeddino_vl instead
from groundeddino_vl.utils.misc import NestedTensor
from groundeddino_vl.data.transforms import Compose
from groundeddino_vl.models.grounding_dino import GroundingDINO
```

## Common Migration Patterns

### Pattern 1: Basic Inference (No Changes Needed)

```python
# Old code (still works, shows deprecation warning)
from groundingdino import load_model, predict, load_image

model = load_model(config_path, checkpoint_path)
image = load_image(image_path)
result = predict(model, image, "cat . dog")

# New code (recommended)
from groundeddino_vl import load_model, predict, load_image

model = load_model(config_path, checkpoint_path)
image = load_image(image_path)
result = predict(model, image, "cat . dog")
```

### Pattern 2: Using Internal Utilities (Requires Changes)

```python
# Old code (no longer works)
from groundingdino.util.misc import NestedTensor
from groundingdino.util.box_ops import box_cxcywh_to_xyxy

# New code (required)
from groundeddino_vl.utils.misc import NestedTensor
from groundeddino_vl.utils.box_ops import box_cxcywh_to_xyxy
```

### Pattern 3: Using Data Transforms (Requires Changes)

```python
# Old code (no longer works)
from groundingdino.datasets.transforms import Compose, RandomResize, ToTensor

# New code (required)
from groundeddino_vl.data.transforms import Compose, RandomResize, ToTensor
```

### Pattern 4: Building Models (Requires Changes)

```python
# Old code (no longer works)
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig

# New code (required)
from groundeddino_vl.models import build_model
from groundeddino_vl.utils.slconfig import SLConfig
```

### Pattern 5: CUDA Extension (Requires Changes)

```python
# Old code (no longer works)
import groundingdino._C as _C

# New code (required)
import groundeddino_vl._C as _C
```

## Automated Migration Script

Use this bash script to automatically update your Python files:

```bash
#!/bin/bash
# migrate_to_groundeddino_vl.sh

# Replace common import patterns
find . -name "*.py" -type f -exec sed -i \
  -e 's/from groundingdino\.util\./from groundeddino_vl.utils./g' \
  -e 's/from groundingdino\.datasets\./from groundeddino_vl.data./g' \
  -e 's/from groundingdino\.models\./from groundeddino_vl.models./g' \
  -e 's/import groundingdino\.util/import groundeddino_vl.utils/g' \
  -e 's/import groundingdino\.datasets/import groundeddino_vl.data/g' \
  -e 's/import groundingdino\.models/import groundeddino_vl.models/g' \
  -e 's/import groundingdino\._C/import groundeddino_vl._C/g' \
  {} \;

echo "Migration complete! Review changes before committing."
```

**Important:** Always review the automated changes before committing them to ensure correctness.

## Module Name Changes

Note these namespace differences when migrating:

| Old (groundingdino) | New (groundeddino_vl) |
|---------------------|----------------------|
| `util` | `utils` |
| `datasets` | `data` |
| `models.GroundingDINO` | `models.grounding_dino` |

## Frequently Asked Questions

### Q: Do I need to migrate immediately?

**A:** No. If you're using the high-level API (`load_model`, `predict`, etc.), your code will continue to work with deprecation warnings. However, if you're importing internal modules, you'll need to migrate.

### Q: Will the groundingdino namespace be removed?

**A:** The compatibility shim will be maintained for the foreseeable future, but new features will only be added to `groundeddino_vl`. We recommend migrating when convenient.

### Q: Can I use both namespaces in the same project?

**A:** Yes, but it's not recommended. The compatibility shim re-exports from `groundeddino_vl`, so you're essentially using the same code. Stick to one namespace for consistency.

### Q: What about my existing model weights?

**A:** Model weights are fully compatible. You can load weights trained with the old namespace using the new namespace and vice versa.

### Q: How do I suppress the deprecation warnings?

**A:** While we recommend migrating, you can suppress the warnings temporarily:

```python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="groundingdino")

import groundingdino  # No warning shown
```

### Q: I'm getting ImportError after migrating. What should I check?

**A:** Common issues:
1. Check for the namespace rename: `util` → `utils`, `datasets` → `data`
2. Ensure you're using `groundeddino_vl` (with underscore), not `groundeddino-vl` (with hyphen)
3. Make sure you have the latest version installed: `pip install --upgrade groundeddino-vl`

## Getting Help

If you encounter issues during migration:

1. Check this migration guide for common patterns
2. Review the [API Reference](https://github.com/ghostcipher1/GroundedDINO-VL#readme)
3. Open an issue on [GitHub](https://github.com/ghostcipher1/GroundedDINO-VL/issues)

## Summary

- **High-level API**: Continues to work via compatibility shim (with warnings)
- **Internal modules**: Must migrate to `groundeddino_vl`
- **Deprecation timeline**: Compatibility shim maintained indefinitely, but migrate when possible
- **Migration effort**: Simple find-replace for most code

The migration is straightforward and can be done incrementally. Start by migrating internal module imports, then update high-level API usage at your convenience.
