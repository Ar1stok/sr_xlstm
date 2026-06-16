"""Main training script for ASR model using Hydra configuration."""
import logging

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer
from safetensors.torch import load_file

from data.data_manager import load_from_parquet
from data.iterable_dataset import ZipAudioDataset
from trainer import TrainerxLSTM
from utils.metrics import call_compute_wer
from utils.utils import compute_mel_stats, compute_mfcc_stats

logger = logging.getLogger(__name__)

ZIP_FOLDER = "/storage0/kpa/datasets/speech/sova/RuYouTube"
ZIP_LIST = [
    "part_0.zip", "part_1.zip",
    "part_2.zip", "part_3.zip",
    "part_4.zip", "part_5.zip",
    "part_6.zip", "part_7.zip",
    "part_8.zip", "part_9.zip",
    "part_10.zip", "part_11.zip",
    "part_12.zip", "part_13.zip",
    "part_14.zip", "part_15.zip",
    "part_16.zip", "part_17.zip",
    "part_18.zip", "part_19.zip",
    "part_20.zip", "part_21.zip",
    "part_22.zip", "part_23.zip",
    "part_24.zip", "part_25.zip",
    "part_26.zip", "part_27.zip",
    "part_28.zip", "part_29.zip",
    "part_30.zip", "part_31.zip",
    "part_32.zip", "part_33.zip",
    "part_34.zip", "part_35.zip",
]

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Runs the main training workflow."""
    # Log configuration.
    yaml_config = OmegaConf.to_yaml(cfg)
    logger.info("\n%s", yaml_config)
    checkpoint = cfg.get("checkpoint_dir", None)

    # Load tokenizer and update vocab size.
    tokenizer = AutoTokenizer.from_pretrained(cfg.datasets.tokenizer_path)
    vocab_size = len(tokenizer)
    cfg.datasets.vocab_size = vocab_size

    blank_token = cfg.datasets.blank_token
    blank_id = tokenizer.convert_tokens_to_ids(blank_token)
    logger.info(f"Tokenizer init")

    # Load Datasets
    train_dataset = load_from_parquet(path=cfg.datasets.train_dataset_path)
    logger.info(f"train_dataset init")

    eval_dataset = load_from_parquet(path=cfg.datasets.valid_dataset_path)
    logger.info(f"eval_dataset init")

    # Initialize mean & std.
    mfcc_mean = torch.tensor([cfg.datasets.MFCC_GLOBAL_MEAN])
    mfcc_std = torch.tensor([cfg.datasets.MFCC_GLOBAL_STD])
    logger.info(f"mfcc_mean: {mfcc_mean}, mfcc_std: {mfcc_std}")

    # Instantiate model, training arguments, and data collator.
    model = hydra.utils.instantiate(
        cfg.models,
        num_classes=vocab_size,
        mean_global=mfcc_mean,
        std_global=mfcc_std,
    )

    # Load State Model
    state = load_file(f"{checkpoint}/model.safetensors")
    model.load_state_dict(state, strict=True)

    training_args = hydra.utils.instantiate(cfg.trainer.training_args)
    data_collator = hydra.utils.instantiate(cfg.trainer.data_collator)

    # Setup metrics.
    compute_metrics = call_compute_wer(tokenizer, blank_id=blank_id)

    # Initialize trainer.
    lr_args = cfg.trainer.lr_scheduler
    trainer = TrainerxLSTM(
        model=model,
        lr_args=lr_args,
        blank_id=blank_id,
        debug=cfg.debug,
        alpha=cfg.alpha,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )

    # Save final config.
    cfg_path = hydra.utils.to_absolute_path("used_config.yaml")
    OmegaConf.save(cfg, cfg_path)
    logger.info("Saved config to %s", cfg_path)

    # Test on one Batch
    batch = next(iter(trainer.get_train_dataloader()))
    logger.info(f"Batch keys: {batch.keys()}")
    logger.info(f"Shapes: {batch['input_values'].shape}") 
    logger.info(f"Batch: {batch}")

    # Run training and evaluation (resume_from_checkpoint=checkpoint).
    trainer.train()
    trainer.evaluate()
    trainer.save_model()


if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
