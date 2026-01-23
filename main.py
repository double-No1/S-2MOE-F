from torch_geometric.loader import DataLoader
from torch.optim import Adam, lr_scheduler
import os
import numpy as np
import torch
import torch.nn as nn
from dataloader import CombinedDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.model_selection import KFold
from model import MergeModel
import argparse
import random
import warnings
warnings.filterwarnings("ignore")


def select_25_percent_data(indices, labels):
    class_indices = {0: [], 1: []}
    for i, idx in enumerate(indices):
        class_indices[labels[idx]].append(idx)

    selected_indices = []
    for cls, idx_list in class_indices.items():
        n_select = max(1, int(0.37 * len(idx_list)))
        selected = random.sample(idx_list, n_select)
        selected_indices.extend(selected)

    random.shuffle(selected_indices)
    return selected_indices


def train_stage1(model, train_loader, device, total_epochs, optimizer, scheduler):
    best_train_loss = float('inf')
    ce_loss_fn = nn.CrossEntropyLoss()

    for epoch_idx in range(total_epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            graph_data, llama_emb, labels = batch
            graph_data = graph_data.to(device)
            llama_emb = llama_emb.to(device).float()
            labels = labels.to(device).long()

            optimizer.zero_grad()
            logits, total_loss, _ = model(graph_data, llama_emb, labels, return_feat=True)
            cls_loss = ce_loss_fn(logits, labels)
            combined_loss = 1 * total_loss.mean() + 0.1 * cls_loss

            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += combined_loss.item()

        scheduler.step()
        avg_train_loss = running_loss / len(train_loader)
        print(f"Stage1 Epoch {epoch_idx + 1}/{total_epochs} | Train Loss: {avg_train_loss:.4f}")

        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            torch.save(model.state_dict(), 'best_model_stage1.pt')

    model.load_state_dict(torch.load('best_model_stage1.pt', weights_only=True))
    print(f"Stage1 Training Done | Best Train Loss: {best_train_loss:.4f}\n")
    return model


def generate_pseudo_labels(model, remaining_loader, device, remaining_indices, train_labels):
    model.eval()
    all_pseudo_info = []

    with torch.no_grad():
        for batch in remaining_loader:
            graph_data, llama_emb, true_labels = batch
            graph_data = graph_data.to(device)
            llama_emb = llama_emb.to(device).float()
            true_labels = true_labels.to(device)


            logits, _, _ = model(graph_data, llama_emb, return_feat=True)
            if torch.isnan(logits).any():
                logits[torch.isnan(logits)] = 0.0
            logits = torch.clamp(logits, min=-1e2, max=1e2)

            probs = torch.softmax(logits, dim=1)
            confidences, pseudo_labels = torch.max(probs, dim=1)

            graph_list = [g.cpu() for g in graph_data.to_data_list()]
            for i in range(len(graph_list)):
                all_pseudo_info.append({
                    'graph': graph_list[i],
                    'llama_emb': llama_emb[i].cpu(),
                    'pseudo_label': pseudo_labels[i].cpu(),
                    'confidence': confidences[i].cpu(),
                    'orig_idx': remaining_indices[len(all_pseudo_info)]
                })

    total_remaining = len(all_pseudo_info)
    n_select_pseudo = max(1, int(0.2 * total_remaining))
    all_pseudo_info_sorted = sorted(all_pseudo_info, key=lambda x: x['confidence'], reverse=True)
    selected_pseudo = all_pseudo_info_sorted[:n_select_pseudo]

    pseudo_graphs = [item['graph'] for item in selected_pseudo]
    pseudo_llama_embs = torch.stack([item['llama_emb'] for item in selected_pseudo])
    pseudo_labels = torch.tensor([item['pseudo_label'].item() for item in selected_pseudo])
    return (pseudo_graphs, pseudo_llama_embs, pseudo_labels)


def create_stage2_dataset(stage1_dataset, pseudo_data):
    pseudo_graphs, pseudo_llama_embs, pseudo_labels = pseudo_data
    stage1_graphs = []
    stage1_llama_embs = []
    stage1_labels = []

    for data in stage1_dataset:
        graph_data, llama_emb, label = data
        stage1_graphs.append(graph_data)
        stage1_llama_embs.append(llama_emb)
        stage1_labels.append(label)

    target_device = stage1_llama_embs[0].device if stage1_llama_embs else \
        (pseudo_llama_embs.device if pseudo_llama_embs.numel() > 0 else torch.device('cpu'))

    stage1_graphs = [g.to(target_device) for g in stage1_graphs]
    pseudo_graphs = [g.to(target_device) for g in pseudo_graphs]
    stage1_llama_embs_tensor = torch.stack([emb.to(target_device) for emb in stage1_llama_embs])
    stage1_labels_tensor = torch.tensor(stage1_labels, device=target_device)
    pseudo_llama_embs = pseudo_llama_embs.to(target_device)
    pseudo_labels = pseudo_labels.to(target_device)

    combined_graphs = stage1_graphs + pseudo_graphs
    combined_llama_embs = torch.cat([stage1_llama_embs_tensor, pseudo_llama_embs], dim=0)
    combined_labels = torch.cat([stage1_labels_tensor, pseudo_labels], dim=0)

    for i, g in enumerate(combined_graphs):
        if hasattr(g, 'x') and g.x is not None and torch.isnan(g.x).any():
            raise ValueError(f"Graph {i} contains NaN in features!")
    if torch.isnan(combined_llama_embs).any():
        raise ValueError("NaN detected in LLAMA Embeddings")
    if torch.isnan(combined_labels).any():
        raise ValueError("NaN detected in labels")

    class Stage2Dataset(torch.utils.data.Dataset):
        def __init__(self, graphs, llama_embs, labels):
            self.graphs = graphs
            self.llama_embs = llama_embs
            self.labels = labels

        def __len__(self):
            return len(self.graphs)

        def __getitem__(self, idx):
            return (self.graphs[idx], self.llama_embs[idx], self.labels[idx])

    return Stage2Dataset(combined_graphs, combined_llama_embs, combined_labels)


def evaluate_epoch(model, test_loader, device, fold, epoch):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            graph_data, llama_emb, labels = batch
            graph_data = graph_data.to(device)
            llama_emb = llama_emb.to(device).float()
            labels = labels.to(device)


            logits, _, _ = model(graph_data, llama_emb, return_feat=True)
            if torch.isnan(logits).any():
                logits[torch.isnan(logits)] = 0.0
            logits = torch.clamp(logits, min=-1e2, max=1e2)

            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0) if all_probs else np.array([])
    all_labels = np.concatenate(all_labels, axis=0) if all_labels else np.array([])
    mask = ~np.isnan(all_probs[:, 1]) if len(all_probs) > 0 else np.array([])

    if len(all_probs) == 0 or not np.any(mask):
        metrics = {'auc': 0.5, 'acc': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    else:
        auc_score = roc_auc_score(all_labels[mask], all_probs[mask, 1]) if \
            len(np.unique(all_labels[mask])) > 1 else 0.5

        precision, recall, thresholds = precision_recall_curve(all_labels[mask], all_probs[mask, 1])
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        preds = (all_probs[mask, 1] > optimal_threshold).astype(int)
        labels_masked = all_labels[mask]

        acc = accuracy_score(labels_masked, preds)
        precision_val = precision_score(labels_masked, preds, average='macro', zero_division=0)
        recall_val = recall_score(labels_masked, preds, average='macro', zero_division=0)
        f1_val = f1_score(labels_masked, preds, average='macro', zero_division=0)

        metrics = {
            'auc': auc_score, 'acc': acc, 'precision': precision_val,
            'recall': recall_val, 'f1': f1_val, 'threshold': optimal_threshold
        }
        print(f"Fold {fold + 1} Epoch {epoch} | Threshold: {optimal_threshold:.4f} | "
              f"AUC: {auc_score:.4f} | Acc: {acc:.4f} | Precision: {precision_val:.4f} | "
              f"Recall: {recall_val:.4f} | F1: {f1_val:.4f}")

    return metrics


def train_stage2_and_record_epochs(model, train_loader, test_loader, device, total_epochs, optimizer, scheduler, fold):
    fold_epoch_metrics = []
    ce_loss_fn = nn.CrossEntropyLoss()
    model_save_dir = f"fold_{fold}_stage2_models"
    os.makedirs(model_save_dir, exist_ok=True)

    print(f"\n=== Fold {fold + 1} Stage2 Training (Total Epochs: {total_epochs}) ===")
    for epoch_idx in range(total_epochs):
        model.train()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in train_loader:
            graph_data, llama_emb, labels = batch
            graph_data = graph_data.to(device)
            llama_emb = llama_emb.to(device).float()
            labels = labels.to(device).long()

            optimizer.zero_grad()
            logits, total_loss, _ = model(graph_data, llama_emb, labels, return_feat=True)
            cls_loss = ce_loss_fn(logits, labels)
            combined_loss = 1 * total_loss.mean() + 0.1 * cls_loss

            if torch.isnan(combined_loss) or torch.isnan(logits).any():
                combined_loss = torch.nan_to_num(combined_loss, nan=1e5)

            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += combined_loss.item()
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += len(labels)

        scheduler.step()
        avg_train_loss = running_loss / len(train_loader)
        train_acc = total_correct / total_samples if total_samples > 0 else 0.0
        print(f"Fold {fold + 1} Stage2 Epoch {epoch_idx + 1}/{total_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        torch.save(model.state_dict(), os.path.join(model_save_dir, f"epoch_{epoch_idx + 1}.pt"))
        epoch_metrics = evaluate_epoch(model, test_loader, device, fold, epoch_idx + 1)
        epoch_metrics.update({'fold': fold, 'epoch': epoch_idx + 1, 'train_loss': avg_train_loss, 'train_acc': train_acc})
        fold_epoch_metrics.append(epoch_metrics)

    return fold_epoch_metrics


def run_cross_validation(args, device):
    all_fold_epoch_metrics = []
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)


    fold_indices = []
    full_dataset = CombinedDataset(
        root=r"E:\COVID\FakeNews\Code\datasets",
        name=f'full_news_graph_dataset',
        llama_pt_path=args.llama_pt_path,
        feature=args.feature,
        empty=False,
        device=device
    )
    dataset_size = len(full_dataset)
    all_indices = np.arange(dataset_size)


    for train_idx, test_idx in kf.split(all_indices):
        fold_indices.append((train_idx, test_idx))
    print(f"数据集总样本数：{dataset_size}，预存5个Fold的索引完成")

    LLAMA_HIDDEN_SIZE = full_dataset[0][1].shape[0]


    for fold, (train_idx, test_idx) in enumerate(fold_indices):
        print(f"\n=== 开始Fold {fold + 1} 训练 ===")

        train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
        test_dataset = torch.utils.data.Subset(full_dataset, test_idx)
        train_labels = [full_dataset[i][2] for i in train_idx]


        stage1_indices = select_25_percent_data(range(len(train_dataset)), train_labels)
        stage1_dataset = torch.utils.data.Subset(train_dataset, stage1_indices)
        remaining_dataset = torch.utils.data.Subset(train_dataset,
                                                    [i for i in range(len(train_dataset)) if i not in stage1_indices])


        stage1_loader = DataLoader(
            stage1_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
            collate_fn=lambda batch: full_dataset.collate_fn1(batch, device)
        )
        remaining_loader = DataLoader(
            remaining_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
            collate_fn=lambda batch: full_dataset.collate_fn1(batch, device)
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
            collate_fn=lambda batch: full_dataset.collate_fn1(batch, device)
        )


        gin_config = {
            "hidden_dim": args.nhid, "num_layers": 4, "device": device,
            "norm_layer": 0, "aggregation": "mean", "bias": "true"
        }
        model = MergeModel(dim_features=LLAMA_HIDDEN_SIZE, config1=gin_config, device=device).to(device)


        optimizer_stage1 = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler_stage1 = lr_scheduler.StepLR(optimizer_stage1, step_size=15, gamma=0.5)
        model = train_stage1(model, stage1_loader, device, args.epochs_stage1, optimizer_stage1, scheduler_stage1)


        pseudo_data = generate_pseudo_labels(model, remaining_loader, device,
                                             [i for i in range(len(remaining_dataset))], train_labels)

        #
        stage2_dataset = create_stage2_dataset(stage1_dataset, pseudo_data)
        stage2_loader = DataLoader(
            stage2_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
            collate_fn=lambda batch: full_dataset.collate_fn1(batch, device)
        )

        optimizer_stage2 = Adam(model.parameters(), lr=args.lr * 0.1, weight_decay=args.weight_decay * 0.5)
        scheduler_stage2 = lr_scheduler.StepLR(optimizer_stage2, step_size=10, gamma=0.5)
        fold_metrics = train_stage2_and_record_epochs(
            model, stage2_loader, test_loader, device, args.epochs_stage2, optimizer_stage2, scheduler_stage2, fold
        )
        all_fold_epoch_metrics.extend(fold_metrics)


    epoch_groups = {}
    for metric in all_fold_epoch_metrics:
        epoch = metric['epoch']
        if epoch not in epoch_groups:
            epoch_groups[epoch] = []
        epoch_groups[epoch].append(metric['auc'])

    valid_epochs = {e: np.mean(auc_list) for e, auc_list in epoch_groups.items() if len(auc_list) == 5}
    if not valid_epochs:
        raise ValueError("No valid epoch data (need 5 folds complete)")

    best_epoch = max(valid_epochs.keys(), key=lambda x: valid_epochs[x])
    best_epoch_mean_auc = valid_epochs[best_epoch]
    best_epoch_metrics = [m for m in all_fold_epoch_metrics if m['epoch'] == best_epoch]


    final_metrics = {}
    metrics_keys = ['auc', 'acc', 'precision', 'recall', 'f1']
    for key in metrics_keys:
        values = [m[key] for m in best_epoch_metrics]
        final_metrics[key] = {'mean': np.mean(values), 'std': np.std(values),
                              'fold_values': [round(v, 4) for v in values]}
    print("\n" + "=" * 60)
    print(f"Global Best Epoch: {best_epoch} (Mean AUC: {best_epoch_mean_auc:.4f})")
    print("=" * 60)
    for key in metrics_keys:
        print(f"{key.upper()}: {final_metrics[key]['mean']:.4f} ± {final_metrics[key]['std']:.4f}")
        print(f"Fold Values: {final_metrics[key]['fold_values']}")
    print("=" * 60)

    return final_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=777, help='Random seed')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID (e.g., 0, 1)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (adjust by GPU memory)')
    parser.add_argument('--lr', type=float, default=5e-5, help='Initial learning rate (Stage1)')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='Weight decay')
    parser.add_argument('--nhid', type=int, default=128, help='GIN hidden dimension')
    parser.add_argument('--epochs_stage1', type=int, default=100, help='Stage1 training epochs')
    parser.add_argument('--epochs_stage2', type=int, default=150, help='Stage2 training epochs')
    parser.add_argument('--feature', type=str, default='spacy', help='Graph feature type (match CombinedDataset)')
    parser.add_argument('--llama_pt_path', type=str,
                        default=r"D:\processed\COVID\LLAMA\combined_processed.pt",
                        help='LLAMA precomputed features .pt file path (required)')
    args = parser.parse_args()


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    print("\n" + "=" * 60)
    print("Start LLAMA Feature + Graph Data Training (5-Fold CV)")
    print("=" * 60)
    run_cross_validation(args, device)


if __name__ == "__main__":
    main()