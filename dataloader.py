import os
import os.path as osp
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data, Batch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
from torch_geometric.utils import add_self_loops, coalesce
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import KFold


def create_graph(text, label, user_id, post_id, tokenizer=None, bert_model=None, device='cpu', llama_emb=None):
    if llama_emb is not None:

        text_feature = llama_emb.unsqueeze(0)
        feat_return = llama_emb
    else:

        if not text:
            text = "empty text"
        inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=128
        ).to(device)

        with torch.no_grad():
            outputs = bert_model(**inputs)
        text_feature = outputs.last_hidden_state.mean(dim=1)
        feat_return = (inputs['input_ids'], inputs['attention_mask'])

    node_ids = [post_id, user_id]
    if len(set(node_ids)) < 2:
        raise ValueError(f"post_id and user_id are duplicated: {post_id} and {user_id}")
    node_id_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
    num_nodes = len(node_ids)

    edge_index = [
        [node_id_to_idx[post_id], node_id_to_idx[user_id]],
        [node_id_to_idx[user_id], node_id_to_idx[post_id]]
    ]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    if edge_index.dim() == 1:
        edge_index = edge_index.view(2, -1)

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"Initial edge index shape is abnormal! Expected [2, E], actual {edge_index.shape} (value={edge_index.tolist()})"
        )

    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    edge_index, _ = coalesce(edge_index, num_nodes=num_nodes)

    if edge_index.dim() == 1:
        edge_index = edge_index.view(2, -1)

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"Final edge index shape is abnormal! Expected [2, E], actual {edge_index.shape} (value={edge_index.tolist()})"
        )

    node_features = torch.cat([text_feature, text_feature], dim=0).to(device)  # [2, hidden_dim]

    if not isinstance(label, int) or label < 0:
        raise ValueError(f"Label must be a non-negative integer, actual is {label} (type {type(label)})")

    graph_data = Data(
        x=node_features,
        edge_index=edge_index,
        y=torch.tensor([label], dtype=torch.long, device=device),
        filename=f"{post_id}.json"
    )
    graph_data.post_id = post_id
    graph_data.user_id = user_id

    return graph_data, feat_return


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer=None, max_length=128, device='cpu'):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

        if len(self.texts) != len(self.labels):
            raise ValueError(f"texts length={len(texts)} does not match labels length={len(labels)}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index out of bounds! Requested {idx}, total samples {len(self)}")

        text = self.texts[idx]
        label = self.labels[idx]

        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=self.max_length
            ).to(self.device)
            return {k: v.squeeze(0) for k, v in encoding.items()}, torch.tensor(label, dtype=torch.long,
                                                                                device=self.device)

        return text, torch.tensor(label, dtype=torch.long, device=self.device)


class CombinedDataset(InMemoryDataset):
    def __init__(self, root, name, bert_path=None, feature='bert', texts=None, labels=None,
                 user_ids=None, post_ids=None, empty=False, transform=None,
                 pre_transform=None, pre_filter=None, fold_idx=0, device='cpu',
                 llama_pt_path=None):
        self.name = name
        self.root = root
        self.feature = feature

        self.texts = texts if texts is not None else []
        self.labels = labels if labels is not None else []
        self.user_ids = user_ids if user_ids is not None else [f'user_{i}' for i in range(len(self.texts))]
        self.post_ids = post_ids if post_ids is not None else [f'post_{i}' for i in range(len(self.texts))]

        self.bert_path = bert_path
        self.llama_pt_path = llama_pt_path
        self.fold_idx = fold_idx
        self.device = device

        self.original_size = len(self.texts)
        self.llama_dict = None

        if self.llama_pt_path and os.path.exists(self.llama_pt_path):
            self.llama_dict = self._load_llama_features()
            print(
                f"Loaded Qwen precomputed features: {len(self.llama_dict)} samples | Feature dimension: {self.llama_dict[next(iter(self.llama_dict.keys()))][0].shape[0]}")
        else:

            if self.bert_path is None:
                raise ValueError(
                    "bert_path (Qwen model path) not passed, and llama_pt_path does not exist or is not passed")

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.bert_path,
                    local_files_only=True,
                    padding_side="right",
                    trust_remote_code=True
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                self.bert_model = AutoModel.from_pretrained(
                    self.bert_path,
                    local_files_only=True,
                    trust_remote_code=True
                ).to(self.device).eval()

            except Exception as e:
                raise RuntimeError(f"Qwen model/tokenizer loading failed (path: {self.bert_path}): {e}")

        if len(self.texts) != len(self.labels):
            raise ValueError(f"texts={len(self.texts)} length does not match labels={len(self.labels)} length")
        if len(self.texts) != len(self.user_ids):
            raise ValueError(f"texts={len(self.texts)} length does not match user_ids={len(self.user_ids)} length")
        if len(self.texts) != len(self.post_ids):
            raise ValueError(f"texts={len(self.texts)} length does not match post_ids={len(self.post_ids)} length")

        super().__init__(root, transform, pre_transform, pre_filter)

        print(
            f"Initialized dataset: original sample count={self.original_size}, using fold {fold_idx + 1}, device={self.device}")

        if not empty:
            if os.path.exists(self.processed_paths[0]):
                try:
                    data_tuple = torch.load(self.processed_paths[0])
                    self.data, self.slices = data_tuple[0], data_tuple[1]
                    self.folds = data_tuple[2]

                    if len(data_tuple) >= 4:
                        self.llama_dict = data_tuple[3]

                    self._set_fold_indices(fold_idx)
                    self._check_sample_count()
                    self._check_indices()
                    self._check_all_graphs()
                    print(f"Successfully loaded processed data: sample count={len(self)}")
                except Exception as e:
                    print(f"Data loading failed: {e}, reprocessing...")
                    self.process()
            else:
                self.process()

    def _load_llama_features(self):
        try:
            data = torch.load(self.llama_pt_path, weights_only=True)

            required_keys = ["embeddings", "labels", "filenames"]
            for key in required_keys:
                if key not in data:
                    raise KeyError(f"Qwen feature file missing key field: {key}")
            llama_dict = {}
            for emb, label, fname in zip(data["embeddings"], data["labels"], data["filenames"]):
                llama_dict[fname] = (emb.float().to(self.device), int(label))
            return llama_dict
        except Exception as e:
            raise RuntimeError(f"Qwen feature loading failed (path: {self.llama_pt_path}): {e}") from e

    def _set_fold_indices(self, fold_idx):
        if fold_idx >= len(self.folds):
            raise ValueError(f"Requested fold index {fold_idx} is out of range, total folds {len(self.folds)}")
        self.train_idx, self.test_idx = self.folds[fold_idx]
        print(f"Using fold {fold_idx + 1}: training samples {len(self.train_idx)}, test samples {len(self.test_idx)}")

    def _check_sample_count(self):
        processed_size = len(self)
        if processed_size != self.original_size:
            raise RuntimeWarning(
                f"Sample count inconsistent: original {self.original_size} → processed {processed_size}, possible filtering"
            )

    def _check_indices(self):
        dataset_size = len(self)
        if dataset_size == 0:
            return
        for idx_name, indices in [
            ("Training", self.train_idx),
            ("Testing", self.test_idx)
        ]:
            if indices.numel() == 0:
                continue
            max_idx = indices.max().item()
            min_idx = indices.min().item()
            if max_idx >= dataset_size:
                raise IndexError(f"{idx_name} index out of bounds: max {max_idx} ≥ sample count {dataset_size}")
            if min_idx < 0:
                raise IndexError(f"{idx_name} index contains negative value: {min_idx}")

    def _check_all_graphs(self):
        dataset_size = len(self)
        if dataset_size == 0:
            return
        print(f"Verifying edge index validity of all graph data (total {dataset_size} samples)...")
        for i in range(dataset_size):
            if i % 100 == 0:
                print(f"Verified {i}/{dataset_size} samples")
            graph = self[i][0]
            num_nodes = graph.num_nodes
            edge_index = graph.edge_index
            filename = getattr(graph, 'filename', f'post_{i}.json')
            if edge_index.ndim != 2:
                raise RuntimeError(
                    f"Graph data {i} (filename={filename}) edge index dimension is abnormal! "
                    f"Expected 2D, actual {edge_index.ndim}D (value={edge_index.tolist()})"
                )
            if edge_index.shape[0] != 2:
                raise RuntimeError(
                    f"Graph data {i} (filename={filename}) edge index shape is abnormal! "
                    f"Expected first dimension 2, actual {edge_index.shape[0]} (shape={edge_index.shape})"
                )
            if edge_index.numel() > 0:
                max_idx = edge_index.max().item()
                min_idx = edge_index.min().item()
                if max_idx >= num_nodes or min_idx < 0:
                    raise RuntimeError(
                        f"Graph data {i} (filename={filename}) index is invalid! "
                        f"Node count={num_nodes}, edge index range [{min_idx}, {max_idx}]"
                    )
            if self.llama_dict and filename not in self.llama_dict:  # Key modification: llama_dict → qwen_dict
                raise RuntimeError(
                    f"Graph data {i} (filename={filename}) has no corresponding Qwen feature! "
                    f"Please check if filename is included in Qwen feature file"
                )
        print("Edge index verification passed for all graph data: dimensions and ranges are valid")

    def _split_data(self):
        dataset_size = len(self)
        if dataset_size < 5:
            print(
                "Insufficient samples (less than 5), cannot perform 5-fold cross-validation, using leave-one-out method")
            n_splits = dataset_size
        else:
            n_splits = 5
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        self.folds = []
        all_indices = np.arange(dataset_size)
        for train_idx, test_idx in kf.split(all_indices):
            self.folds.append((
                torch.tensor(train_idx, dtype=torch.long),
                torch.tensor(test_idx, dtype=torch.long)
            ))
        self._set_fold_indices(self.fold_idx)

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['placeholder.txt']

    @property
    def processed_file_names(self):
        return f'{self.name}_data_{self.feature}.pt'

    def download(self):
        os.makedirs(self.raw_dir, exist_ok=True)
        placeholder_path = osp.join(self.raw_dir, 'placeholder.txt')
        if not os.path.exists(placeholder_path):
            with open(placeholder_path, 'w', encoding='utf-8') as f:
                f.write('Used to pass PyTorch Geometric raw file check')

    def process(self):
        print(f"Start processing data: original sample count={self.original_size}")
        if not self.llama_dict and self.original_size == 0:  # Key modification: llama_dict → qwen_dict
            raise ValueError("No input samples (texts is empty), and no valid Qwen precomputed feature path provided")
        if self.llama_dict and len(self.llama_dict) == 0:  # Key modification: llama_dict → qwen_dict
            raise ValueError("Qwen precomputed features are empty, please check feature file content")
        data_list = []
        feat_list = []
        process_count = len(self.llama_dict) if self.llama_dict else self.original_size
        for i in range(process_count):
            try:
                if self.llama_dict:
                    filename = list(self.llama_dict.keys())[i]
                    llama_emb, llama_label = self.llama_dict[filename]
                    post_id = filename.replace(".json", "")
                    if i < len(self.user_ids):
                        user_id = self.user_ids[i]
                    else:
                        user_id = f'user_{i}'
                    text = self.texts[i] if (i < len(self.texts) and self.texts) else f"auto_text_{i}"
                    graph_data, feat = create_graph(
                        text=text,
                        label=llama_label,
                        user_id=user_id,
                        post_id=post_id,
                        device=self.device,
                        llama_emb=llama_emb
                    )
                else:
                    text = self.texts[i] if i < len(self.texts) else ""
                    label = self.labels[i] if i < len(self.labels) else 0
                    user_id = self.user_ids[i] if i < len(self.user_ids) else f'user_{i}'
                    post_id = self.post_ids[i] if i < len(self.post_ids) else f'post_{i}'
                    graph_data, feat = create_graph(
                        text=text,
                        label=label,
                        user_id=user_id,
                        post_id=post_id,
                        tokenizer=self.tokenizer,
                        bert_model=self.bert_model,
                        device=self.device
                    )
                num_nodes = graph_data.num_nodes
                edge_index = graph_data.edge_index
                if edge_index.numel() > 0:
                    max_idx = edge_index.max().item()
                    min_idx = edge_index.min().item()
                    if max_idx >= num_nodes or min_idx < 0:
                        raise ValueError(
                            f"Graph data final check failed! Sample {i} (post_id={post_id}) "
                            f"node count={num_nodes}, edge index min={min_idx}, max={max_idx}"
                        )

                data_list.append(graph_data)
                feat_list.append(feat)

            except Exception as e:
                raise RuntimeError(f"Sample {i} (post_id={post_id}) processing failed: {e}") from e

        if len(data_list) == 0:
            raise RuntimeError("No valid graph data generated, please check input or feature file")
        print(f"Successfully generated {len(data_list)} graph data")
        self.data, self.slices = InMemoryDataset.collate(data_list)
        self._split_data()
        save_data = (self.data, self.slices, self.folds)
        if self.llama_dict:  # Key modification: llama_dict → qwen_dict
            save_data += (self.llama_dict,)  # Key modification: llama_dict → qwen_dict
        torch.save(save_data, self.processed_paths[0])
        print(f"Data saving completed: {self.processed_paths[0]}")

    def __getitem__(self, idx):
        dataset_size = len(self)
        if idx < 0 or idx >= dataset_size:
            raise IndexError(f"Sample index out of bounds: requested {idx}, total samples {dataset_size}")
        graph = super().__getitem__(idx)
        graph = graph.to(self.device)
        if self.llama_dict:
            filename = getattr(graph, 'filename', f'post_{idx}.json')
            _, label = self.llama_dict[filename]
        else:
            label = self.labels[idx] if idx < len(self.labels) else 0
        if self.llama_dict:
            filename = getattr(graph, 'filename', f'post_{idx}.json')
            feat, _ = self.llama_dict[filename]
        else:
            text = self.texts[idx] if idx < len(self.texts) else ""
            post_id = self.post_ids[idx] if idx < len(self.post_ids) else f'post_{idx}'
            _, feat = create_graph(
                text=text,
                label=label,
                user_id=self.user_ids[idx] if idx < len(self.user_ids) else f'user_{idx}',
                post_id=post_id,
                tokenizer=self.tokenizer,
                bert_model=self.bert_model,
                device=self.device
            )
        return graph, feat, label

    def __len__(self):
        return len(self.slices['y']) - 1 if 'y' in self.slices else 0


def collate_fn1(batch, device='cpu', use_llama=False):
    if len(batch) == 0:
        raise ValueError("Batch input is empty")
    graphs = [item[0] for item in batch]
    feats = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    batch_size = len(batch)
    try:
        graph_batch = Batch.from_data_list(graphs).to(device)
    except Exception as e:
        raise RuntimeError(f"Graph data merging failed: {e}") from e
    if graph_batch.edge_index.ndim != 2 or graph_batch.edge_index.shape[0] != 2:
        raise RuntimeError(
            f"Merged graph edge index is abnormal! Shape={graph_batch.edge_index.shape} (expected [2, total_E])"
        )
    if use_llama:
        try:
            feat_batch = torch.stack(feats, dim=0).to(device)
        except Exception as e:
            raise RuntimeError(f"Qwen feature merging failed: {e}") from e
        feature_return = feat_batch

    else:
        input_ids_list = [f[0].squeeze(0) for f in feats]
        attention_mask_list = [f[1].squeeze(0) for f in feats]
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=0).to(device)
        attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0).to(device)
        feature_return = (input_ids, attention_mask)

    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    if labels_tensor.shape[0] != batch_size:
        raise ValueError(f"Label shape is abnormal: expected {batch_size} samples, actual {labels_tensor.shape[0]}")
    return graph_batch, feature_return, labels_tensor
