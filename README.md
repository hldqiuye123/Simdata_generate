# Simdata_generate
📦 Data Loading
﻿
The training and validation dataloaders are created using the `make_train_data` and `make_eval_data` functions defined in [`dataset.py`](./dataset.py). Specifically:
﻿
```python
train_loader = dataset.make_train_data(args)
val_loader = dataset.make_eval_data(args)
```
﻿
These functions are responsible for constructing the corresponding `torch.utils.data.DataLoader` objects according to the input arguments specified in `args`. The expected fields include dataset path, batch size, data augmentation options, and other preprocessing parameters.
