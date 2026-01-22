"""Main training script for ASR model using Hydra configuration."""
import logging

import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from data.data_manager import load_from_parquet
from trainer import TrainerxLSTM
from utils.metrics import call_compute_wer
from utils.utils import compute_mel_stats

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Runs the main training workflow."""
    # Log configuration.
    yaml_config = OmegaConf.to_yaml(cfg)
    logger.info("\n%s", yaml_config)

    # Load tokenizer and update vocab size.
    tokenizer = AutoTokenizer.from_pretrained(cfg.datasets.tokenizer_path)
    vocab_size = len(tokenizer)
    cfg.datasets.vocab_size = vocab_size

    blank_token = cfg.datasets.blank_token
    blank_id = tokenizer.convert_tokens_to_ids(blank_token)

    # Load datasets.
    dataset = load_from_parquet(path=cfg.datasets.dataset_path)
    train_dataset = dataset["train"].select(
        range(cfg.datasets.train_n_examples)
    )
    eval_dataset = dataset["valid"].select(
        range(cfg.datasets.eval_n_examples)
    )

    # Compute mel spectrogram statistics.
    mel_mean, mel_std = compute_mel_stats(
        train_dataset, max_items=cfg.datasets.mel_stats_max_items
    )
    cfg.datasets.mel_mean = mel_mean
    cfg.datasets.mel_std = mel_std
    logger.info("Mel stats: mean=%.3f, std=%.3f", mel_mean, mel_std)

    # Instantiate model, training arguments, and data collator.
    model = hydra.utils.instantiate(
        cfg.models,
        num_classes=vocab_size,
        mel_mean=mel_mean,
        mel_std=mel_std,
    )

    training_args = hydra.utils.instantiate(cfg.trainer.training_args)
    data_collator = hydra.utils.instantiate(cfg.trainer.data_collator)

    # Setup metrics.
    compute_metrics = call_compute_wer(tokenizer, blank_id=blank_id)

    # Initialize trainer.
    trainer = TrainerxLSTM(
        model=model,
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

    # Run training and evaluation.
    trainer.train()
    trainer.evaluate()
    trainer.save_model()


if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
