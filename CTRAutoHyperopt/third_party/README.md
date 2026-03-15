# Third Party Dependencies

This directory contains vendored third-party libraries for CTR Auto HyperOpt.

## Included Libraries

### DeepCTR-Torch
- **Source**: https://github.com/shenweichen/DeepCTR-Torch
- **License**: Apache-2.0
- **Version**: Latest (cloned 2026-03-15)
- **Description**: Deep Learning CTR Models in PyTorch

### MLGB (Machine Learning Great Boss)
- **Source**: https://github.com/UlionTse/mlgb
- **License**: Apache-2.0
- **Version**: Latest (cloned 2026-03-15)
- **Description**: 50+ CTR Prediction & Recommender System Models

## Usage

These libraries are included directly to ensure version compatibility and avoid dependency conflicts.

To use them in your code:

```python
import sys
sys.path.insert(0, 'third_party/DeepCTR-Torch')
sys.path.insert(0, 'third_party/mlgb')

from deepctr_torch.models import DeepFM, xDeepFM
from mlgb.torch.models.ranking import MLP, DCN
```

## Updates

To update these libraries:

```bash
cd third_party
rm -rf DeepCTR-Torch mlgb
git clone --depth 1 https://github.com/shenweichen/DeepCTR-Torch.git
git clone --depth 1 https://github.com/UlionTse/mlgb.git
rm -rf DeepCTR-Torch/.git mlgb/.git
```
