# Deep-Learning
Graph Attention Networks (GAT)
============

- Paper link: [https://arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903)
- Author's code repo (in Tensorflow):
  [https://github.com/PetarV-/GAT](https://github.com/PetarV-/GAT).
- Popular pytorch implementation:
  [https://github.com/Diego999/pyGAT](https://github.com/Diego999/pyGAT).

Dependencies
------------
- torch v1.0: the autograd support for sparse mm is only available in v1.0.
- requests
- sklearn

```bash
pip install torch==1.0.0 requests
```

How to run
----------

Run with following:

```bash
python train.py --dataset=cora --gpu=0
```

```bash
python train.py --dataset=citeseer --gpu=0
```

```bash
python train.py --dataset=pubmed --gpu=0 --num-out-heads=8 --weight-decay=0.001
```

```bash
python train_ppi.py --gpu=0
```