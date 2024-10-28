---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python [conda env:ProSmi]
  language: python
  name: python3
---

+++ {"deletable": true, "editable": true, "frozen": false}

# Code running tutorial

+++ {"deletable": true, "editable": true, "frozen": false}

## Environment building

+++ {"deletable": true, "editable": true, "frozen": false}

### Build the runtime environment code

+++ {"deletable": true, "editable": true, "frozen": false}

```bash
mamba env create -f env.yaml
```

+++ {"deletable": true, "editable": true, "frozen": false}

### Large model training weights download code

+++ {"deletable": true, "editable": true, "frozen": false}

```bash
huggingface-cli download --resume-download DeepChem/ChemBERTa-77M-MTR --local-dir code/preprocessing/DeepChem/ChemBERTa-77M-MTR 

```

+++ {"deletable": true, "editable": true, "frozen": false}

## Training process

+++ {"deletable": true, "editable": true, "frozen": false, "jp-MarkdownHeadingCollapsed": true}

```{important}
Ensure that all samples to be tested have been pre-processed here and the embedding information has been extracted
```

+++ {"deletable": true, "editable": true, "frozen": false}

### Preprocessing: Extracting embedding information

+++ {"deletable": true, "editable": true, "frozen": false}

```python
python code/preprocessing/preprocessing.py --train_val_path data/training_data/ESP/train_val_yb \
                                                               --outpath data/training_data/ESP/embeddings \
                                                               --smiles_emb_no 2000 --prot_emb_no 2000
```

+++ {"deletable": true, "editable": true, "frozen": false}

### Train a large model

+++ {"deletable": true, "editable": true, "frozen": false}

```python
CUDA_VISIBLE_DEVICES=1 python code/training/training.py --train_dir data/training_data/ESP/train_val_yb/BAHD_ESP_train_df.csv \
                                --val_dir data/training_data/ESP/train_val_yb/BAHD_ESP_val_df.csv \
                                --embed_path data/training_data/ESP/embeddings \
                                --save_model_path data/training_data/ESP/saved_model \
                                --pretrained_model data/training_data/BindingDB/saved_model/pretraining_IC50_6gpus_bs144_1.5e-05_layers6.txt.pkl \
                                --learning_rate 1e-5  --num_hidden_layers 6 --batch_size 24 --binary_task True \
                                --log_name ESP --num_train_epochs 100 --port 12558
```                               

+++ {"deletable": true, "editable": true, "frozen": false}

### Train a gradient boosting tree

+++ {"deletable": true, "editable": true, "frozen": false}

```python
CUDA_VISIBLE_DEVICES=1 python code/training/training_GB.py --train_dir data/training_data/ESP/train_val_yb/BAHD_ESP_train_df.csv \
                                --val_dir data/training_data/ESP/train_val_yb/BAHD_ESP_val_df.csv \
                                --test_dir data/training_data/ESP/train_val_yb/T1_ESP_test_df.csv \
                                --pretrained_model data/training_data/ESP/saved_model/ESP_2gpus_bs48_1e-05_layers6.txt.pkl \
                                --embed_path data/training_data/ESP/embeddings \
                                --save_xgb_path data/training_data/ESP/saved_model/xgb \
                                --save_pred_path data/training_data/ESP/saved_predictions \
                                --num_hidden_layers 6 --num_iter 500 --log_name ESP --binary_task True		
```                               

+++ {"deletable": true, "editable": true, "frozen": false}

## Independent test code

+++ {"deletable": true, "editable": true, "frozen": false, "jp-MarkdownHeadingCollapsed": true}

```{important}
Ensure that all samples to be tested have been pre-processed here and the embedding information has been extracted
```

+++ {"deletable": true, "editable": true, "frozen": false}

```python
CUDA_VISIBLE_DEVICES=1 python code/training/prediction.py \
    --test_dir data/prediction_data/T1_ESP_test_df.csv \
    --pretrained_model data/training_data/ESP/saved_model/ESP_2gpus_bs48_1e-05_layers6.txt.pkl \
    --embed_path data/training_data/ESP/embeddings \
    --xgb_path data/training_data/ESP/saved_model/xgb \
    --save_pred_path data/training_data/ESP/saved_predictions_yb \
    --num_hidden_layers 6 \
    --num_iter 500 \
    --log_name ESP \
    --binary_task True	
```
