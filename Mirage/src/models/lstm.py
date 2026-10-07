"""LSTM that reads an image as a sequence of raw pixel patches. There are no convolutional layers."""
import torch
import torch.nn.functional as F
from torch import nn

# LSTMs on long sequences can produce very large gradients, train.py clips to this norm
GRAD_CLIP_NORM = 1.0


class PatchLSTM(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, hidden_size, num_layers, dropout):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size {img_size} must be divisible by patch_size {patch_size}")
        self.patch_size = patch_size
        self.seq_len = (img_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2
        # one linear layer shrinks each flattened patch, embed_dim=None feeds raw patches to the LSTM
        self.embed = nn.Linear(patch_dim, embed_dim) if embed_dim else nn.Identity()
        self.lstm = nn.LSTM(embed_dim or patch_dim, hidden_size, num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(2 * hidden_size, 1)
        self.grad_clip_norm = GRAD_CLIP_NORM

    def patchify(self, x):
        """(B, 3, H, W) -> (B, number of patches, 3 * patch * patch), patches in row-major order."""
        return F.unfold(x, self.patch_size, stride=self.patch_size).transpose(1, 2)

    def forward(self, x):
        seq = self.embed(self.patchify(x))
        _, (h, _) = self.lstm(seq)
        # h is (layers * 2, B, hidden): the last two entries are the top layer's forward
        # and backward final states
        return self.head(torch.cat([h[-2], h[-1]], dim=1)).squeeze(1)


def build(img_size=224, patch_size=16, embed_dim=128, hidden_size=128, num_layers=2, dropout=0.2):
    """The sequence length is (img_size / patch_size) ** 2, so it is set through img_size and patch_size."""
    return PatchLSTM(img_size, patch_size, embed_dim, hidden_size, num_layers, dropout)
