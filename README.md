Attention-based Graph Neural Network
============
Related Works
------------
- GIN Paper Link: [https://arxiv.org/abs/1810.00826](https://arxiv.org/abs/1810.00826)
- GAT Paper Link: [https://arxiv.org/abs/1807.07984](https://arxiv.org/abs/1807.07984)
- GAT Pytorch Tutorial Link: [https://docs.dgl.ai/tutorials/models/1_gnn/9_gat.html](https://docs.dgl.ai/tutorials/models/1_gnn/9_gat.html)


Library Needed
------------
- torch
- dgl
- numpy


Running Samples
----------


```bash
python train.py --dataset=cora --gpu=0
```

```bash
python train.py --dataset=citeseer --gpu=0
```

```bash
python train.py --dataset=pubmed --gpu=0 --num-out-heads=8 --weight-decay=0.001
```