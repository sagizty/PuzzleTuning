## MoCo v3 for Self-supervised ResNet and ViT

The original repo of MoCo-v3 could be found [here](https://github.com/facebookresearch/moco-v3)

Pip requirements:	timm == 0.4.9, PyTorch == 1.9.0, Torchvision == 0.10.0, Cuda == 10.2, Numpy == 1.19

Typical BASH: 
   ```console
python main_moco.py \
  -a vit_base -b 512\
  --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
  --epochs=100 --warmup-epochs=20 \
  --stop-grad-conv1 --moco-m-cos --moco-t=.2 --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed --world-size 1 --rank 0 --basic_state_dict the/path/of/CPIA
   ```
