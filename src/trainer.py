"""Trainers for Wav2Vec2CTC and ASRxLSTM models"""
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TrainerxLSTM(Trainer):
    """Custom Trainer for xLSTM models with CTC loss and noise injection.

    Extends HuggingFace Trainer for xLSTM ASR models. Supports Gaussian noise
    injection during training and comprehensive debug logging.
    Expects inputs from DataCollatorASRWithPadding with precomputed lengths.
    """
    def __init__(
            self,
            *args,
            lr_args,
            blank_id,
            alpha: float = 0.0,
            debug: bool = False,
            **kwargs
        ) -> None:
        """Initialize xLSTM Trainer with CTC loss.

        Args:
            blank_id: CTC blank token ID.
            alpha: Noise injection scale (0.0 = disabled).
            debug: Enable debug logging if True.
            *args: Arguments for parent Trainer.
            **kwargs: Keyword arguments for parent Trainer.
        """
        super().__init__(*args, **kwargs)
        self.lr_args = lr_args
        self.debug = debug
        self.alpha = alpha
        self.blank_id = blank_id
        self.ctc_loss = nn.CTCLoss(
            reduction="mean", zero_infinity=True, blank=blank_id
        )

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None
    ):
        """Compute CTC loss for xLSTM model with optional noise injection.

        Args:
            model: xLSTM model returning logits dict.
            inputs: Batch from DataCollatorASRWithPadding containing:
                - input_values: Audio features
                - input_ids: Labels with -100 padding
                - input_lengths: Audio sequence lengths
                - targets_lengths: Target sequence lengths
                - attention_mask: Optional attention mask
            return_outputs: Return loss and logits if True.
            num_items_in_batch: Number of items (unused).

        Returns:
            Loss tensor or tuple (loss, outputs).
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        input_values = inputs["input_values"].to(device)
        labels = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        input_lengths = inputs["input_lengths"].to(device)
        target_lengths = inputs["targets_lengths"].to(device)

        # Forward pass
        outputs = model(input_values,
                        attention_mask=attention_mask,
                        labels=labels)
        logits = outputs["logits"]  # (B, T, C)

        # Optional noise injection during training
        if model.training and self.alpha > 0:
            logits = logits + self.alpha * torch.randn_like(logits)

        # Prepare concatenated targets for CTC
        targets = []
        for i in range(labels.shape[0]):
            tgt = labels[i][labels[i] != -100]
            targets.append(tgt)
        targets = torch.cat(targets).long()

        if self.debug:
            logger.info(
                "blank_id=%d, pad_id=%d", 
                self.blank_id, self.processing_class.pad_token_id
            )
            logger.info(
                "blank_token=%s", 
                self.processing_class.convert_ids_to_tokens(self.blank_id)
            )
            logger.info("CTC DEBUG batch_size=%d", labels.size(0))

            for i in range(labels.size(0)):
                raw = labels[i].tolist()
                clean = [t for t in raw if t != -100]
                logger.info(
                    "CTC DEBUG sample %d: raw_len=%d, \
                    clean_len=%d, target_length=%d",
                    i, len(raw), len(clean), target_lengths[i].item()
                )
            logger.info(
                "CTC DEBUG total targets len=%d, sum(target_lengths)=%d",
                targets.numel(), target_lengths.sum().item()
            )

        # CTC loss computation: (B, T, C) -> (T, B, C)
        log_probs = F.log_softmax(
            logits,
            dim=-1,
            dtype=torch.float32
        ).transpose(0, 1)
        loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)

        if self.debug and self.state.global_step % 100 == 0:
            with torch.no_grad():
                pred_ids = logits.argmax(dim=-1)  # (B, T)
                unique, counts = torch.unique(pred_ids, return_counts=True)
                logger.info(
                    "TRAIN PRED DEBUG step=%d, unique_ids=%s, counts=%s",
                    self.state.global_step, unique.tolist(), counts.tolist()
                )

        outputs = {"logits": logits}
        return (loss, outputs) if return_outputs else loss


    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys: Optional[list] = None,
    ):
        """Perform prediction step during evaluation.

        Args:
            model: Model instance.
            inputs: Batch inputs.
            prediction_loss_only: Compute only loss if True.
            ignore_keys: Keys to ignore in outputs.

        Returns:
            Tuple of (loss, logits, labels) or (loss, None, None).
        """
        model.eval()
        inputs = self._prepare_inputs(inputs)

        if prediction_loss_only:
            loss = self.compute_loss(model, inputs, return_outputs=False)[0]
            return (loss, None, None)

        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        loss = loss.detach() if loss is not None else None
        logits = outputs["logits"].detach()
        labels = inputs.get("input_ids")
        if labels is not None:
            labels = labels.detach()

        return (loss, logits, labels)
    
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        # 1. Setup the optimizer
        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(self.model)
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                    "weight_decay": 0.0,
                },
            ]
            # Используем стандартный AdamW из PyTorch
            self.optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                eps=self.args.adam_epsilon,
            )

        # Setup CyclicLR if it True in cfg
        if self.lr_scheduler is None:
            if self.lr_args.cyclic:
                self.lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
                    self.optimizer,
                    base_lr=self.lr_args.base_lr,
                    max_lr=self.args.learning_rate,
                    step_size_up=self.lr_args.step_size_up,
                    step_size_down=self.lr_args.step_size_down,
                    mode=self.lr_args.mode,
                    cycle_momentum=False,
                )
            else:
                from transformers.optimization import get_scheduler
                
                self.lr_scheduler = get_scheduler(
                    name=self.args.lr_scheduler_type,
                    optimizer=self.optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                )