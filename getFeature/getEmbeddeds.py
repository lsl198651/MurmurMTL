import torch
import torch.nn as nn


class SparseFeature:
    """The Feature Class for Sparse feature.

    Args:
        name (str): feature's name.
        vocab_size (int): vocabulary size of embedding table.
        embed_dim (int): embedding vector's length
        shared_with (str): the another feature name which this feature will shared with embedding.
        padding_idx (int, optional): If specified, the entries at padding_idx will be masked 0 in InputMask Layer.
        initializer(Initializer): Initializer the embedding layer weight.
    """

    def __init__(
            self, name, vocab_size, embed_dim=16, shared_with=None, padding_idx=None
    ):
        self.name = name
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.shared_with = shared_with
        self.padding_idx = padding_idx

    def __repr__(self):
        return (
            f"<SparseFeature {self.name} with Embedding shape ({self.vocab_size},"
            f" {self.embed_dim})>"
        )

    def get_embedding_layer(self):
        if not hasattr(self, "embed"):
            self.embed = torch.nn.Embedding(int(self.vocab_size), self.embed_dim)
        return self.embed


class EmbeddingLayer(nn.Module):
    """General Embedding Layer.
    We save all the feature embeddings in embed_dict: `{feature_name : embedding table}`.
    Args:
        features (list): the list of `Feature Class`. It is means all the features which we want to create a embedding table.
    Shape:
        - Input:
            x (dict): {feature_name: feature_value}, sequence feature value is a 2D tensor with shape:`(batch_size, seq_len)`,\
                      sparse/fc1 feature value is a 1D tensor with shape `(batch_size)`.
            features (list): the list of `Feature Class`. It is means the current features which we want to do embedding lookup.
            squeeze_dim (bool): whether to squeeze dim of output (default = `False`).
        - Output:
            - if input Dense: `(batch_size, num_features_dense)`.
            - if input Sparse: `(batch_size, num_features, embed_dim)` or  `(batch_size, num_features * embed_dim)`.
            - if input Sequence: same with input sparse or `(batch_size, num_features_seq, seq_length, embed_dim)` when `pooling=="concat"`.
            - if input Dense and Sparse/Sequence: `(batch_size, num_features_sparse * embed_dim)`. Note we must squeeze_dim for concat fc1 value with sparse embedding.
    """

    def __init__(self, features):
        super().__init__()
        self.features = features
        self.embed_dict = nn.ModuleDict()
        self.n_dense = 0

        for fea in features:
            if fea.name in self.embed_dict:  # exist
                continue
            if isinstance(fea, SparseFeature) and fea.shared_with == None:
                self.embed_dict[fea.name] = fea.get_embedding_layer()

    def forward(self, x, features, squeeze_dim=False):
        sparse_emb = []
        sparse_exists = False

        for fea in features:
            if isinstance(fea, SparseFeature):
                sparse_emb.append(self.embed_dict[fea.name](x[fea.name].long()).unsqueeze(1))
            else:
                raise ValueError(
                    "If keep the original shape:[batch_size, num_features, embed_dim],"
                    " expected %s in feature list, got %s"
                    % ("SparseFeatures", features)
                )

        if len(sparse_emb) > 0:
            sparse_exists = True
            sparse_emb = torch.cat(sparse_emb, dim=1)  # [batch_size, num_features, embed_dim]
        # Note: if the emb_dim of sparse features is different, we must squeeze_dim
        if (squeeze_dim):
            if sparse_exists:
                # squeeze dim to : [batch_size, num_features*embed_dim]
                return sparse_emb.flatten(start_dim=1)
            else:
                raise ValueError("The input features can note be empty")
        else:
            if sparse_exists:

                return sparse_emb
            else:
                raise ValueError(
                    "If keep the original shape:[batch_size, num_features, embed_dim],"
                    " expected %s in feature list, got %s"
                    % ("SparseFeatures", features)
                )
