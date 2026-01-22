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


class TrainerWav2Vec(Trainer):
    """Custom Trainer for Wav2Vec2 models with CTC loss computation.

    Extends HuggingFace Trainer to handle CTC loss for Wav2Vec2 models.
    Supports debug logging and frozen backbone during training.
    Expects inputs from DataCollatorCTCWithPadding.
    """
    def __init__(
        self,
        *args,
        debug: bool = False,
        **kwargs
    ) -> None:
        """Initialize Wav2Vec2 Trainer with CTC loss.

        Args:
            debug: Enable debug logging if True.
            *args: Arguments for parent Trainer.
            **kwargs: Keyword arguments for parent Trainer.
        """
        super().__init__(*args, **kwargs)
        self.ctc_loss = nn.CTCLoss(reduction="mean", zero_infinity=True)
        self.debug = debug

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None
    ):
        """Compute CTC loss for Wav2Vec2 model.

        Args:
            model: Wav2Vec2 model.
            inputs: Batch inputs containing:
                - input_values: Audio features (batch, seq_len)
                - input_ids: Token labels (batch, target_len) 
                  with -100 for padding
                - attention_mask: Optional attention mask (batch, seq_len)
            return_outputs: Return loss and logits if True.
            num_items_in_batch: Number of items (unused).

        Returns:
            Loss tensor or tuple (loss, outputs) with logits.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        input_values = inputs["input_values"].to(device)
        labels = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")

        if self.debug:
            logger.info("input_values shape: %s", input_values.shape)
            logger.info("labels shape: %s", labels.shape)

        # Forward pass with optional frozen backbone
        with (
            torch.no_grad() if model.training
            and hasattr(model, "freeze_backbone")
            and model.freeze_backbone else torch.enable_grad()
        ):
            logits = model(input_values, attention_mask=attention_mask)

        if self.debug:
            logger.info("logits shape: %s", logits.shape)

        # Prepare CTC inputs
        input_lengths = torch.full(
            (logits.shape[0],), logits.shape[1], dtype=torch.long
        )

        # Replace -100 with pad_token_id for CTC
        labels_masked = labels.clone()
        labels_masked[labels == -100] = self.processor.tokenizer.pad_token_id
        target_lengths = (
            labels_masked != self.processor.tokenizer.pad_token_id
        ).sum(-1)

        if self.debug:
            logger.info("input_lengths: %s",
                        input_lengths)
            logger.info("target_lengths: %s",
                        target_lengths)
            logger.info("pad_token_id: %d",
                        self.processor.tokenizer.pad_token_id)

        # Compute CTC loss: logits -> (T, B, C)
        log_probs = (F.log_softmax(logits, dim=-1, dtype=torch.float32)
                    .transpose(0, 1))
        loss = self.ctc_loss(
            log_probs,
            labels_masked,
            input_lengths,
            target_lengths
        )

        if self.debug:
            logger.info("CTC loss: %.4f", loss.item())

        outputs = {"logits": logits}
        return (loss, outputs) if return_outputs else loss


    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict,
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
            loss = self.compute_loss(model, inputs, return_outputs=False)
            return (loss, None, None)

        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        loss = loss.detach() if loss is not None else None
        logits = outputs["logits"].detach()
        labels = inputs.get("input_ids")
        if labels is not None:
            labels = labels.detach()

        return (loss, logits, labels)


class TrainerxLSTM(Trainer):
    """Custom Trainer for xLSTM models with CTC loss and noise injection.

    Extends HuggingFace Trainer for xLSTM ASR models. Supports Gaussian noise
    injection during training and comprehensive debug logging.
    Expects inputs from DataCollatorASRWithPadding with precomputed lengths.
    """
    def __init__(
            self,
            *args,
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
