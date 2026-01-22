"""WER metric computation for CTC ASR evaluation."""
import logging

import numpy as np
import torch
from jiwer import wer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def call_compute_wer(tokenizer, blank_id: int = 2):
    """Create WER metric function for HuggingFace Trainer.

    Performs CTC decoding (remove duplicates + blanks), converts predictions
    and labels to text, computes Word Error Rate using jiwer.

    Args:
        tokenizer: Tokenizer with decode() and pad_token_id attributes.
        blank_id: CTC blank token ID (typically 0, 2, or vocab_size-1).
    """
    def compute_metrics(eval_pred) -> dict[str, float]:
        """Compute Word Error Rate from model logits and labels.

        Args:
            eval_pred: Tuple of (logits: np.ndarray, labels: np.ndarray).

        Returns:
            Dict with 'wer' key containing word error rate [0.0, inf).
        """
        logger.info('Computing WER metrics...')
        logits, labels = eval_pred
        logger.info(
            'Logits shape: %s, labels shape: %s', 
            logits.shape, labels.shape)

        # Convert logits to predictions via argmax.
        logits_torch = torch.from_numpy(logits).to(torch.float32)
        pred_ids = torch.argmax(logits_torch, dim=-1)  # [batch, time]

        # Debug first 3 samples.
        for i in range(min(3, pred_ids.shape[0])):
            path = pred_ids[i].cpu().numpy()
            unique_ids, counts = np.unique(path, return_counts=True)
            logger.info(
                '[PRED DEBUG] sample %d: unique_ids=%s, counts=%s, first_30=%s',
                i, unique_ids, counts, path[:30])

        # CTC decoding: collapse duplicates + filter blanks.
        pred_texts = []
        for path in pred_ids:
            # torch.unique_consecutive removes consecutive duplicates.
            collapsed = torch.unique_consecutive(path, dim=0)
            # Filter out blank tokens.
            tokens = [token.item() for token in collapsed if token != blank_id]
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            pred_texts.append(text)

        # Process references: replace -100 with pad_token_id, filter pads.
        labels[labels == -100] = tokenizer.pad_token_id  # -100 → pad
        ref_texts = []
        for label in labels:
            # Remove padding tokens.
            tokens = label[label != tokenizer.pad_token_id].tolist()
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            ref_texts.append(text)

        # Compute WER.
        wer_score = wer(ref_texts, pred_texts)
        logger.info('WER: %.4f (batch_size=%d)', wer_score, len(pred_texts))

        if pred_texts:
            logger.info('Sample PRED: %s', pred_texts[0])
            logger.info('Sample REF:  %s', ref_texts[0])

        return {'wer': wer_score}

    return compute_metrics
