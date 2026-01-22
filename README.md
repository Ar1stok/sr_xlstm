# Speech Recognition with xLSTM

This repository contains an experimental pipeline based on Russian speech and an xLSTM model with CTC training. It uses the PyTorch + Hugging Face Transformers + Hydra + DVC stack for reproducing experiments, managing configurations, and managing data.

## Capability

- An xLSTM-based acoustic model for log-mel spectrograms.
- Training with CTC loss, support for custom tokenizers (Hugging Face) and `<blank>` tokens.
- Configuration via Hydra.
- DVC support for dataset versioning and training artifacts.

## Minimal Installation
Create a conda environment from the file `environment_sr_xlstm.yaml`.

Clone repository from github:
```
git clone https://github.com/Ar1stok/sr_xlstm.git
cd sr_xlstm
```

Create conda environment:
```
conda env create -n xlstm -f environment_sr_xlstm.yaml
conda activate xlstm
```

## Work with data
The dataset is stored as .parquet files and managed via DVC.
For using this data follow that commands:
```
pip install "dvc[gdrive]"
dvc pull
```
URL based in `.dvc/config`.

## Training model
For training with based parameters (Standard parameters are used only for the simple test.)
```
cd src
python main.py
```
or you can override parameters (example)
```
# Change name and dir
python main.py experiment_name=xlstm_rudevice output_dir=./outputs/xlstm_rudevice

# Change model and trainer parameters
python main.py models.hidden_size=512 trainer.training_args.num_train_epochs=20
```

## TODO
- *Refine and possibly improve the xLSTM model architecture*
- *Implement scheduled learning (learning rate & curriculum schedules)*
- *Introduce speaker‑based dataset splits*
- *Use the SOVA RuYoutube and synthetic datasets for full pretraining* 
- *Define an end‑to‑end training pipeline* 

> [!NOTE]
> Official repository [NX-AI/xLSTM](https://github.com/NX-AI/xlstm)