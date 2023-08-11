# ANT
Implementation of the paper "Adaptable Node Alignment for Relieving Distribution Shift on Graphs"

![](pic.png)

**Step 1**
Generate a dataset with feature shifts:

```bash
python make_dataset.py cora
python make_dataset.py photo
```

**Step 2**
Use pre-trained models to validate performance on 6 backbone models and the Cora and Photo datasets:
```bash
bash run.sh
```
