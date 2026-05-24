"""xLSTM and Wav2Vec2-based ASR models with CTC heads."""
import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
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
    """xLSTM ASR model: MFCC → Conv frontend → xLSTM → CTC head.

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
        slstm_backend: str = 'cuda',
        mean_global: Optional[torch.Tensor] = None,
        std_global: Optional[torch.Tensor] = None,
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
            mfcc_mean: Global mel mean for normalization.
            mfcc_std: Global mel std for normalization.
            slstm_backend: Backend for sLSTM ('cuda', 'vanilla').
            debug: Enable debug logging.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.hidden_size = hidden_size
        if mean_global is not None and std_global is not None:
            self.mean_global = mean_global  # (n_mfcc, 1)
            self.std_global = std_global    # (n_mfcc, 1)
        else:
            self.mean_global = None
            self.std_global = None
        self.debug = debug

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
            add_post_blocks_norm=True,
            slstm_at=[1, 3, 5, 7],
        )
        self.xlstm_stack = xLSTMBlockStack(xlstm_cfg)

        # CTC linear head.
        self.ctc_head = nn.Linear(hidden_size, num_classes)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize MFCC using global statistics.

        Args:
            x: Input tensor (batch_size, time_steps, num_features).

        Returns:
            Normalized tensor.
        """
        if self.mean_global is not None and self.std_global is not None:
            mean = self.mean_global.to(device=x.device, dtype=x.dtype)  # (n_mfcc, 1)
            std = self.std_global.to(device=x.device, dtype=x.dtype)    # (n_mfcc, 1)
            x_norm = (x - mean) / (std + 1e-6)
        else:
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True)
            x_norm = (x - mean) / (std + 1e-6)

        return x_norm

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

       # 1. Global normalization.
        x = self.normalize(input_values)
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
            logger.info('[DEBUG] After LN+dropout:\
                        shape=%s, mean=%.3f, std=%.13f',
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
