# [Reproduce] Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles (ViT)

ECCV 2016: [Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles](https://arxiv.org/abs/1603.09246)

Reproduced based on https://github.com/YangZyyyy/JigsawPuzzlesPytorch

## Changes:
* Replaced AlexNet with a Vision Transformer 
* Enabled DDP and AMP

## Environment:
* python 3.9
* pytorch 2.0.0

## Generate permutation
```bash
python select_permutation.py
```

## Train
```bash
cd scripts
bash run.sh
```