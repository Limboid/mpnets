from mpnets.utils.nn import sparse_threshold
import torch
import torch.nn as nn


class LeakySpikingBucket(nn.Module):

    decay_factor: float = 0.9
    discharge_penalty: float = 1.0
    spiking_amplitude: float = 1.0
    dropout_lambda: float = 0.05
    max_bucket: float = 5.0
    bucket_threshold: float = 1.0
    min_bucket: float = 0.0

    def __init__(self, **hparams):
        super(LeakySpikingBucket, self).__init__()
        self.bucket = None
        self.__dict__.update(hparams)

    def forward(self, input):
        if not self.bucket:
            self.bucket = torch.zeros_like(input)
        if not self.prev:
            self.prev = torch.zeros_like(input)

        # Main algorithm
        self.bucket = (
            self.decay_factor * self.bucket
            + input
            - self.discharge_penalty * torch.abs(self.prev)
        )
        if self.dropout_lambda:
            self.bucket *= torch.exponential(self.dropout_lambda, size=self.bucket.shape)
        Y = self.spiking_amplitude * sparse_threshold(
            self.bucket, alpha=self.bucket_threshold
        )

        self.prev = Y
        return Y
