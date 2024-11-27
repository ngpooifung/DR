# DR

## Installation

```
conda env create --name DR --file env.yml
conda activate DR
install pytorch that fits your machine (pip install torch==1.12.0+cu116 torchvision==0.12.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116)
```

## Training

All configurations can be found in the train.py, for example:
```python
torchrun --standalone --nnodes=1 --nproc_per_node=1  train.py \
              --process main \
              --dir <training data> \
              --test-dir <validation data during training> \
              --output <path to the output folder> \
              --arch <model architecture>  \
              --epochs <training epochs> \
              --lr <learning rate> \
              --batch_size <batch size> \
              --resize <matching smaller edge of images> \
              --dropout <apply model dropout> \
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
              --output <path to the output folder> \
              --finetune <select saved model checkpoints> \
              --test <name of the result file> \
```

## models
I uploaded a list of models that I used to generate results, they are under the models folder. For each type (RDR, VTDR, gradability), eight models are uploaded. 
model suffix:
384/448/484/576: models trained with different image sizes
inception/dense121/efficient: models trained with different architectures
no suffix: the chosen model with resnet50 and image size 512



