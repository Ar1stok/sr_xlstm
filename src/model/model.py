"""xLSTM and Wav2Vec2-based ASR models with CTC heads."""
import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
from xlstm import (FeedForwardConfig, mLSTMBlockConfig, mLSTMLayerConfig,
                   sLSTMBlockConfig, sLSTMLayerConfig, xLSTMBlockStack,
                   xLSTMBlockStackConfig)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ASRxLSTM(nn.Module):
    """xLSTM ASR model: Mel-spectrograms → Conv frontend → xLSTM → CTC head.

    Expects input_values: (batch, time_steps, n_mels). Applies global mel
    normalization, conv1d feature extraction, xLSTM stack, and CTC projection.
    Handles padding masking with finite logit suppression.
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int,
        num_blocks: int,
        num_classes: int,
        num_heads: int = 4,
        context_length: int = 1024,
        dropout: float = 0.1,
        mel_mean: float = 0.0,
        mel_std: float = 1.0,
        slstm_backend: str = 'cuda',
        debug: bool = False,
    ) -> None:
        """Initialize xLSTM ASR model.

        Args:
            num_features: Input mel-spectrogram feature dimension (n_mels).
            hidden_size: Hidden dimension for conv and xLSTM.
            num_blocks: Number of xLSTM blocks.
            num_classes: CTC output vocabulary size.
            num_heads: Number of attention heads in xLSTM.
            context_length: xLSTM context length.
            dropout: Dropout probability.
            mel_mean: Global mel mean for normalization.
            mel_std: Global mel std for normalization.
            slstm_backend: Backend for sLSTM ('cuda', 'vanilla').
            debug: Enable debug logging.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.debug = debug

        # Global mel normalization buffers.
        self.register_buffer('mel_mean', torch.tensor(mel_mean,
                                                      dtype=torch.float32))
        self.register_buffer('mel_std', torch.tensor(mel_std,
                                                     dtype=torch.float32))

        # Conv frontend: (B, n_mels, T) → (B, hidden, T).
        self.conv_frontend = nn.Sequential(
            nn.Conv1d(
                num_features,
                hidden_size // 2,
                kernel_size=3,
                padding=1),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Conv1d(
                hidden_size // 2,
                hidden_size,
                kernel_size=3,
                padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )

        # Feature normalization and dropout.
        self.feature_ln = nn.LayerNorm(hidden_size)
        self.feature_dropout = nn.Dropout(dropout)

        # xLSTM stack configuration.
        xlstm_cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4,
                    qkv_proj_blocksize=4,
                    num_heads=num_heads,
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend=slstm_backend,
                    num_heads=num_heads,
                    conv1d_kernel_size=4,
                    bias_init='powerlaw_blockdependent',
                ),
                feedforward=FeedForwardConfig(
                    proj_factor=1.3,
                    act_fn='gelu',
                ),
            ),
            context_length=context_length,
            num_blocks=num_blocks,
            embedding_dim=hidden_size,
            slstm_at=[1], # sLSTM at block 1.
        )
        self.xlstm_stack = xLSTMBlockStack(xlstm_cfg)

        # CTC linear head.
        self.ctc_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def normalize_mel(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize mel-spectrograms using global statistics.

        Args:
            x: Input tensor (batch_size, time_steps, num_features).

        Returns:
            Normalized tensor.
        """
        return (x - self.mel_mean) / (self.mel_std + 1e-5)

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Forward pass.

        Args:
            input_values: Mel-spectrograms 
              (batch_size, time_steps, num_features).
            attention_mask: Optional (batch_size, time_steps) with one
              for valid frames.
            labels: Optional CTC labels (ignored for inference).
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, torch.Tensor] with 'logits' or raw logits tensor.
        """
        if self.debug:
            logger.info(
                '[DEBUG] Input: shape=%s, range=[%.3f, %.3f]',
                input_values.shape,
                input_values.min().item(),
                input_values.max().item(),
            )

       # 1. Global mel normalization.
        x = self.normalize_mel(input_values)
        if self.debug:
            logger.info('[DEBUG] After norm: mean=%.3f, std=%.3f',
                        x.mean().item(), x.std().item())


        # 2. Conv frontend: (B, T, F) → (B, F, T) → conv → (B, T, hidden).
        x = x.transpose(1, 2)       # (B, F, T)
        x = self.conv_frontend(x)   # (B, hidden, T)
        x = x.transpose(1, 2)       # (B, T, hidden)

        if self.debug:
            logger.info('[DEBUG] After conv: shape=%s, mean=%.3f, std=%.3f',
                        x.shape, x.mean().item(), x.std().item())

        # 3. Feature LayerNorm + dropout.
        x = self.feature_ln(x)
        x = self.feature_dropout(x)

        if self.debug:
            logger.info('[DEBUG] After LN+dropout: \
                        shape=%s, mean=%.3f, std=%.3f',
                        x.shape, x.mean().item(), x.std().item())

        # 4. xLSTM stack.
        xlstm_out = self.xlstm_stack(x)
        if self.debug:
            logger.info('[DEBUG] xLSTM out: shape=%s, mean=%.3f, std=%.3f',
                        xlstm_out.shape, xlstm_out.mean().item(),
                        xlstm_out.std().item())
        # 5. CTC head.
        logits = self.ctc_head(xlstm_out)
        if self.debug:
            logger.info(
                '[DEBUG] Logits: shape=%s, mean=%.3f, std=%.3f',
                logits.shape, logits.mean().item(), logits.std().item(),
            )

        # 6. Mask padding frames (finite suppression for numerical stability).
        if attention_mask is not None:
            mask = (attention_mask == 0).unsqueeze(-1)  # (B, T, 1)
            logits = logits.masked_fill(mask, -1e4)

        if self.debug and attention_mask is not None:
            valid_frames = attention_mask.sum().item()
            logger.info('[DEBUG] Attention_mask: valid_frames=%f', valid_frames)

        # Return logits dict for Trainer compatibility.
        if labels is not None:
            return {'logits': logits}
        return logits


class Wav2Vec2CTC(nn.Module):
    """Wav2Vec2 backbone with CTC head.

    Supports raw waveforms as input. Optional backbone freezing.
    """
    def __init__(
        self,
        model_path: str = 'wav2vec2-russian-model',
        num_classes: int = 40,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
    ) -> None:
        """Initialize Wav2Vec2 CTC model.

        Args:
            model_path: Path to pretrained Wav2Vec2 model.
            num_classes: CTC vocabulary size.
            dropout: Dropout in CTC head.
            freeze_backbone: Freeze Wav2Vec2 parameters.
        """
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained(model_path,
                                                     local_files_only=True)

        if freeze_backbone:
            for param in self.wav2vec.parameters():
                param.requires_grad = False

        hidden_size = self.wav2vec.config.hidden_size

        self.ctc_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ):
        """Forward pass.

        Args:
            input_values: Raw waveforms (batch_size, time_samples).
            attention_mask: Optional (batch_size, time_samples).
            **kwargs: Ignored.

        Returns:
            CTC logits (batch_size, time_samples, num_classes).
        """
        outputs = self.wav2vec(input_values, attention_mask=attention_mask)
        logits = self.ctc_head(outputs.last_hidden_state)
        return logits
