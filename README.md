# DR

## Installation

```
conda env create --name DR --file env.yml
conda activate DR
install pytorch that fits your machine (pip install torch==1.12.0+cu116 torchvision==0.12.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116)
```

## Training

```python
torchrun --standalone --nnodes=1 --nproc_per_node=1  train.py \
              --process main \
              --dir <training data> \
              --test-dir <validation data during training> \
              --output <output folder directory> \
              --arch <model architecture>  \
              --epochs <training epochs> \
              --lr <learning rate> \
              --batch_size <batch size> \
              --resize <matching smaller edge of images> \
              --weight_decay <weight decay> \
              --checkpoints_n_steps <saving model checkpoint every n steps> \
              --weight <training weights> \
```

## Evaluation
```python
torchrun --standalone --nnodes=1 --nproc_per_node=1  train.py \
              --process eval
              --test-dir <validation data during training> \
              --arch <model architecture>  \
              --batch_size <batch size> \
              --resize <matching smaller edge of images> \
              --finetune <select saved model checkpoints> \
```

