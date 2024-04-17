# PuzzleTuning

https://arxiv.org/abs/2311.06712

Pathological image analysis is a crucial field in computer vision. Due to the annotation scarcity in the pathological field, recently, most of the works have leveraged self-supervised learning (SSL) trained on unlabeled pathological images, hoping to mine the representation effectively. However, there are two core defects in current SSL-based pathological pre-training: (1) they do not explicitly explore the essential focuses of the pathological field, and (2) they do not effectively bridge with and thus take advantage of the knowledge from natural images. To explicitly address them, we propose our large-scale PuzzleTuning framework, containing the following innovations. Firstly, we define three task focuses that can effectively bridge knowledge of pathological and natural domain: appearance consistency, spatial consistency, and restoration understanding. Secondly, we devise a novel multiple puzzle restoring task, which explicitly pre-trains the model regarding these focuses. Thirdly, we introduce an explicit prompt-tuning process to incrementally integrate the domain-specific knowledge. It builds a bridge to align the large domain gap between natural and pathological images. Additionally, a curriculum-learning training strategy is designed to regulate task difficulty, making the model adaptive to the puzzle restoring complexity. Experimental results show that our PuzzleTuning framework outperforms the previous state-of-the-art methods in various downstream tasks on multiple datasets.

<img width="1475" alt="fig_concept" src="https://github.com/sagizty/PuzzleTuning/assets/50575108/3014b515-6276-4282-bb57-5ac15f9343ec">
Samples illustrate the focuses and relationships in pathological images. They are pancreatic liquid samples (a and b) and colonic epithelium tissue samples (c and d) of normal (a and c) and cancer conditions (b and d). The patches of them are numbered from 1 to 9. Grouping the patches from each image as a bag, after intermixing patches among them, the three pathological focuses of appearance consistency, spatial consistency, and restoration understanding are highlighted.

<img width="1265" alt="fig_PuzzleTuning_method" src="https://github.com/sagizty/PuzzleTuning/assets/50575108/6b924946-59a0-4dc0-b3e4-750b38a0359a">
Overview of PuzzleTuning. Three steps are designed in PuzzleTuning: 1) Puzzle making, where image batch are divided into bags of patches and fix-position and relation identity are randomly assigned. The relation patches are then in-place shuffled with each other, making up the puzzle state. 2) Puzzle understanding, where puzzles regarding grouping, junction, and restoration relationships are learned by prompt tokens attached to the encoder. Through the prompt tokens, the pathological focuses are explicitly seamed with general vision knowledge. 3) Puzzle restoring, where the decoder restores the relation patches with position patches as hint, under SSL supervision against original images.


# Usage
## pre-trained weights
we have updated the pre-trained weight of PuzzleTuning and all counterparts at

https://drive.google.com/file/d/1-mddejIdCRP5AscnlWAyEcGzfgBIRCSf/view?usp=share_link

## demo with Colab
we have updated a demo for iullustration at 

https://github.com/sagizty/PuzzleTuning/blob/main/PuzzleTuning%20Colab%20Demo.ipynb

## training script
python -m torch.distributed.launch --nproc_per_node=8 --nnodes 1 --node_rank 0 PuzzleTuning.py --DDP_distributed --batch_size 64 --group_shuffle_size 8 --blr 1.5e-4 --epochs 2000 --accum_iter 2 --print_freq 5000 --check_point_gap 100 --input_size 224 --warmup_epochs 100 --pin_mem --num_workers 32 --strategy loop --PromptTuning Deep --basic_state_dict /home/saved_models/ViT_b16_224_Imagenet.pth --data_path /home/datasets/All

## CPIA dataset
https://github.com/zhanglab2021/CPIA_Dataset

# Results
## Comparison
<img width="794" alt="image" src="https://github.com/sagizty/PuzzleTuning/assets/50575108/b4b9bf27-afd0-49ab-a910-60bb0d0b3c7b">
<img width="193" alt="image" src="https://github.com/sagizty/PuzzleTuning/assets/50575108/0dc41248-b556-4a66-bdfa-5f6d49b60877">

## Domain bridging target
<img width="589" alt="image" src="https://github.com/sagizty/PuzzleTuning/assets/50575108/1a81bfa9-525b-4b90-8abe-f1bed9affa48">

## Domain bridging with Puzzles and Prompts
<img width="1178" alt="Screenshot 2023-10-28 at 4 42 31 PM" src="https://github.com/sagizty/PuzzleTuning/assets/50575108/02c17125-9038-47cd-b239-eb738fc4d8cc">
<img width="1148" alt="Screenshot 2023-10-28 at 4 43 02 PM" src="https://github.com/sagizty/PuzzleTuning/assets/50575108/959e3cd0-d5e3-4bff-b592-70a89163e768">

<img width="528" alt="image" src="https://github.com/sagizty/PuzzleTuning/assets/50575108/1bc601c9-cf65-414d-a2a5-4234a81f04ce">

## Curiculum learning
<img width="898" alt="Screenshot 2023-10-28 at 4 43 36 PM" src="https://github.com/sagizty/PuzzleTuning/assets/50575108/a304b83a-9cb8-4757-bd6a-5c5913008d51">

<img width="544" alt="image" src="https://github.com/sagizty/PuzzleTuning/assets/50575108/fbbb9b89-0bff-416a-8485-5e805926ff69">
