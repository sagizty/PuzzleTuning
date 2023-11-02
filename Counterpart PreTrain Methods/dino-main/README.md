# Self-Supervised Vision Transformers with DINO

The original repo of DINO could be found [here](https://github.com/facebookresearch/dino "DINO")

Pip requirements:	timm == 0.4.9, PyTorch == 1.7.1, Torchvision == 0.8.2, Cuda == 11.0

Typical BASH: 
   ```console
python -m torch.distributed.launch \
--nproc_per_node=2 main_dino.py --arch vit_base --batch_size_per_gpu 128 \
--lr 1.5e-4 --epochs 100 --data_path /root/autodl-tmp/All \
--basic_state_dict /root/autodl-tmp/ViT_b16_224_Imagenet.pth \
--num_workers 32 --output_dir the/path/of/CPIA
   ```
