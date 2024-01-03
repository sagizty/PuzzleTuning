# PuzzleTuning

https://arxiv.org/abs/2311.06712

Pathology image analysis is a crucial field in computer vision. Due to the annotation scarcity in the pathology field, recently, most of the work leverages self-supervised learning (SSL) trained on unlabeled pathology images, hoping to mine the main representation automatically. However, there are two core defects in SSL-based pathological pre-training: (1) they do not explicitly explore the essential characteristics of the pathology field, and (2) they do not effectively align and take advantage of the large natural image domain. To explicitly address them, we propose our large-scale Puzzle-Tuning framework, containing the following three innovations. Firstly, we propose three task focuses on aligning pathological and natural domains: appearance consistency, spatial consistency, and misalignment understanding. Secondly, we devise a puzzle making and reconstruction process as a pre-training task to explicitly pre-train the model based on these focuses. Thirdly, for the large domain gap between natural and pathological fields, we introduce an explicit prompt-tuning process to incrementally integrate the domain-specific knowledge with the natural knowledge. Additionally, we design a curriculum-learning training strategy that regulates the task difficulty, making the model fit the complex puzzle restoring adaptively. Experimental results show that our Puzzle-Tuning framework outperforms the previous SOTA methods in various downstream tasks over multiple datasets. 
![Screenshot 2023-10-30 at 11 49 43 AM](https://github.com/sagizty/PuzzleTuning/assets/50575108/ca2ecbc6-fcb9-408c-88ec-18ec77ad99f8)
Overview of PuzzleTuning. Three steps are designed: In step 1 of the puzzle making process, each input batch of images is split into bags of patches sharing the same size for each patch. Then, the position and relation tokens are randomly assigned to the same position across the batch. The relation tokens are independently shuffled among the bags. While the position tokens, which serve as the fixed patches, all stayed in the original locations. The operation augments grouping and semantic relationship within the different groups of tokens, while the relationship within each bag remains unchanged. In step 2 of the puzzle understanding process: The bags are then sent into the backbone encoder to model the grouping, junction and the additional misalignment relationships. After that, the position tokens are replaced with the original input position tokens to achieve the positional hint for the low-level decoder model. During the prompt tuning process, the pathological focuses are explicitly encoded with the backbone’s general vision knowledge. In step 3 of the puzzle restoring process: The decoder model reconstructs the relation tokens and hint. With SSL supervision, the model restores puzzles of the original images.

# Usage
## weights
we have updated the pre-trained weight of PuzzleTuning and all counterparts at

https://drive.google.com/file/d/1-mddejIdCRP5AscnlWAyEcGzfgBIRCSf/view?usp=share_link

## demo
we have updated a demo for iullustration at 

https://github.com/sagizty/PuzzleTuning/blob/main/PuzzleTuning%20Colab%20Demo.ipynb

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
