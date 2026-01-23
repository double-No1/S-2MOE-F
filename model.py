import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from datetime import datetime
from gin_model.Models import OCGIN
from Losses import OCC_loss, InfoNCELoss
from masknet_train import ParametricMaskNet


class MergeModel(nn.Module):
    def __init__(self, dim_features, config1, device,
                 num_chunks=4, threshold=0.5, alpha=0.2, beta=0.5, gamma=0.3):
        super(MergeModel, self).__init__()
        self.device = device
        self.num_chunks = num_chunks
        self.threshold = threshold
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.gin_features = OCGIN(dim_features, config1).to(device)
        self.llama_proj = nn.Linear(
            dim_features,
            config1['hidden_dim'] * config1['num_layers']
        ).to(device)

        self.hidden_dim = config1['hidden_dim']
        self.num_layers = config1['num_layers']
        self.total_dim = self.hidden_dim * self.num_layers
        self.chunk_dim = self.total_dim // self.num_chunks

        self.mask_net = ParametricMaskNet(
            num_segments=self.num_chunks,
            mask_dim=self.chunk_dim
        ).to(device)

        self.classifier = nn.Linear(self.total_dim, 2)
        self.info_nce_loss = InfoNCELoss()
        self.occ_loss = OCC_loss()

        self.center = nn.Parameter(torch.empty(1, self.total_dim), requires_grad=True)
        nn.init.normal_(self.center, mean=0, std=0.1)
        self.center.data = self.center.data.to(device)

        self.tau = 1.0
        self.epsilon = 1e-8
        self.pseudo_label_threshold = 0.95
        self.top_k_ratio = 0.2

        self.policy_net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Softmax(dim=-1)
        ).to(device)

        self.epsilon_start = 0.9
        self.epsilon_end = 0.1
        self.epsilon_decay = 1000
        self.global_step = 0

        self.log_probs = []
        self.rewards = []

    def split_chunks(self, x):
        assert x.size(1) % self.num_chunks == 0
        return x.view(x.size(0), self.num_chunks, -1)

    def compute_covariance_loss(self, x, y):
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)
        cov_x = x.T @ x / (x.size(0) - 1)
        cov_y = y.T @ y / (y.size(0) - 1)
        return F.mse_loss(cov_x, cov_y)

    def dual_attention_transform(self, graph_embed, text_embed):
        g_chunks = self.split_chunks(graph_embed)
        t_chunks = self.split_chunks(text_embed)

        batch_size = g_chunks.size(0)

        mask_matrix = self.mask_net()
        mask_matrix = mask_matrix.unsqueeze(0).expand(batch_size, -1, -1).to(graph_embed.device)

        g_chunks = g_chunks * mask_matrix
        t_chunks = t_chunks * mask_matrix

        attn1 = F.scaled_dot_product_attention(query=g_chunks, key=t_chunks, value=t_chunks)
        attn2 = F.scaled_dot_product_attention(query=t_chunks, key=g_chunks, value=g_chunks)

        g_new = attn1.reshape(attn1.size(0), -1)
        t_new = attn2.reshape(attn2.size(0), -1)

        return g_new, t_new

    def compute_confidence(self, g_new, t_new):
        center_expanded = self.center.expand_as(g_new)
        dist_g = torch.norm(g_new - center_expanded, p=2, dim=1)
        dist_t = torch.norm(t_new - center_expanded, p=2, dim=1)
        delta = torch.abs(dist_g - dist_t) / (dist_g + dist_t + self.epsilon)
        return 1 - delta

    def compute_center(self, dataloader, device):
        self.eval()
        all_embeddings = []

        with torch.no_grad():
            for batch in dataloader:
                graph_data, llama_emb, _ = batch
                graph_data = graph_data.to(device)
                llama_emb = llama_emb.to(device).float()

                g_emb, _ = self.gin_features(graph_data)
                t_emb = self.llama_proj(llama_emb)

                g_emb = F.normalize(g_emb, dim=1)
                t_emb = F.normalize(t_emb, dim=1)

                g_new, t_new = self.dual_attention_transform(g_emb, t_emb)
                combined_emb = (g_new + t_new) / 2
                all_embeddings.append(combined_emb)

        if all_embeddings:
            all_embeddings = torch.cat(all_embeddings, dim=0)
            self.center.data = all_embeddings.mean(dim=0, keepdim=True)

    def generate_pseudo_labels(self, graph_data, llama_emb):
        self.eval()
        with torch.no_grad():
            g_emb, _ = self.gin_features(graph_data)
            t_emb = self.llama_proj(llama_emb)

            g_emb = F.normalize(g_emb, dim=1)
            t_emb = F.normalize(t_emb, dim=1)

            g_new, t_new = self.dual_attention_transform(g_emb, t_emb)
            confidence = self.compute_confidence(g_new, t_new)
            pseudo_labels = (confidence > self.pseudo_label_threshold).long()
        return pseudo_labels, confidence

    def _get_epsilon(self):

        eps = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
              torch.exp(torch.tensor(-self.global_step / self.epsilon_decay, device=self.device)).item()
        self.global_step += 1
        return eps

    def _compute_rl_reward(self, outputs_1, outputs_2, labels):

        true_news_mask = (labels == 1)
        if not torch.any(true_news_mask):
            return torch.tensor([0.5, 0.5], device=self.device)

        h_t = outputs_1[true_news_mask]
        z_t = outputs_2[true_news_mask]
        theta = self.center.expand(h_t.shape[0], -1)
        r_gnn_per_sample = F.cosine_similarity(h_t, theta, dim=1)
        r_llm_per_sample = F.cosine_similarity(z_t, theta,dim=1)

        r_gnn = r_gnn_per_sample.mean()
        r_llm = r_llm_per_sample.mean()

        reward_signal = torch.stack([r_gnn, r_llm])
        return reward_signal.to(self.device)

    def _rl_weight_routing(self, reward_signal):

        eps = self._get_epsilon()
        policy_dist = self.policy_net(reward_signal.unsqueeze(0))  # [1, 2]

        if torch.rand(1, device=self.device).item() < eps:

            weights = torch.ones(2, device=self.device) / 2.0
        else:
            m = torch.distributions.Categorical(policy_dist)
            action = m.sample()
            weights = F.one_hot(action, num_classes=2).float().squeeze(0)
            if weights.sum() < self.epsilon:
                weights = policy_dist.squeeze(0)

        weight_sum = weights.sum() + self.epsilon
        w_gnn = weights[0] / weight_sum
        w_llm = weights[1] / weight_sum
        log_prob = torch.log(policy_dist + self.epsilon).squeeze(0)
        return w_gnn, w_llm, log_prob

    def reset_rl_state(self):

        self.log_probs = []
        self.rewards = []

    def forward(self, data, llama_emb, labels=None, return_feat=False):
        outputs_1, _ = self.gin_features(data)
        outputs_2 = self.llama_proj(llama_emb)

        outputs_1 = F.normalize(outputs_1.float().to(self.device), dim=1)
        outputs_2 = F.normalize(outputs_2.float().to(self.device), dim=1)
        w_gnn = w_llm = 0.5
        log_prob = None

        if labels is not None:
            reward_signal = self._compute_rl_reward(outputs_1, outputs_2, labels)
            w_gnn, w_llm, log_prob = self._rl_weight_routing(reward_signal)
            self.log_probs.append(log_prob)
            self.rewards.append(reward_signal.mean())

        g_new, t_new = self.dual_attention_transform(outputs_1, outputs_2)
        info_nce_loss = self.info_nce_loss(g_new, t_new)

        occ = 0.0
        if labels is not None:
            occ = w_gnn * self.occ_loss(g_new, labels, self.center) + \
                  w_llm * self.occ_loss(t_new, labels, self.center)

        cov_loss = self.compute_covariance_loss(g_new, t_new)
        total_loss = self.alpha * info_nce_loss + \
                     self.beta * occ + \
                     self.gamma * cov_loss


        if self.training and log_prob is not None and len(self.rewards) > 0:
            policy_loss = -log_prob.mean() * self.rewards[-1]
            total_loss += 0.1 * policy_loss

        combined = (g_new + t_new) / 2
        logits = self.classifier(combined)

        if return_feat:
            return logits, total_loss, combined
        else:
            return logits, total_loss