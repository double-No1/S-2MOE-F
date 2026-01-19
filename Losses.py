import torch
import torch.nn as nn
import torch.nn.functional as F


class OCC_loss(nn.Module):
    def __init__(self, margin=1.0):
        super(OCC_loss, self).__init__()
        self.margin = margin

    def forward(self, embeddings, labels, center):

        dist = torch.norm(embeddings - center, p=2, dim=1)  # [B]
        pos_loss = labels * dist.pow(2)
        neg_loss = (1 - labels) * F.relu(self.margin - dist).pow(2)
        loss = (pos_loss + neg_loss).mean()
        return loss





class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, features1, features2):
        """
        Computes the InfoNCE loss.
        Args:
            features1: output of the first model (batch_size, feature_dim)
            features2: output of the second model (batch_size, feature_dim)
        Returns:
            loss: the InfoNCE loss value
        """
        batch_size = features1.size(0)

        similarity_matrix = torch.mm(features1, features2.T) / self.temperature
        mask = torch.eye(batch_size, dtype=torch.bool, device=similarity_matrix.device)
        positives = similarity_matrix[mask].view(batch_size, 1)
        negatives = similarity_matrix[~mask].view(batch_size, -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=similarity_matrix.device)
        loss = F.cross_entropy(logits, labels)

        return loss
