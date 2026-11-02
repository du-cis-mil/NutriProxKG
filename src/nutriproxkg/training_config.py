from dataclasses import dataclass


@dataclass
class E2EConfig:
    """Hyper-parameters for end-to-end transductive nutrition regression."""

    # --- architecture ---
    feature_dim: int = 64  # learnable input embedding size per node
    hidden_dim: int = 128  # branch / fusion hidden width
    head_hidden: int = 64  # regression head hidden width
    dropout: float = 0.4
    n_taxonomy_layers: int = 2  # message-passing rounds on the genus (SS) tree
    n_ii_layers: int = 2  # message-passing rounds on the II graphs
    attn_heads: int = 4  # heads for the gated fusion attention

    # --- compound branch ---
    topk_compounds: int = 256  # per-ingredient cap (top by content, then presence)
    compound_agg: str = "attn"  # "attn" (set-attention) | "mean" (DeepSets ablation)
    freeze_compound: bool = False

    # --- which branches are active (for ablation) ---
    use_compound: bool = True
    use_taxonomy: bool = True
    use_ii_isa: bool = True
    use_ii_sim: bool = True
    # use the similar_to cosine weight as an edge attribute in the II-similarity
    # branch attention (the ontology is_a vs compound-cosine separation is always
    # on via the two distinct relations; this adds the cosine magnitude)
    use_edge_attr: bool = True

    # --- optimisation ---
    epochs: int = 600
    lr: float = 2e-3
    weight_decay: float = 1e-4  # on conv / head params
    emb_weight_decay: float = 1e-3  # stronger decay on the large embedding tables
    patience: int = 60  # early stopping (inner-val regression MSE)
    min_delta: float = 1e-4
    inner_val_frac: float = 0.15  # held out from the train fold for early stopping

    # --- auxiliary self-supervised structural loss (joint end-to-end) ---
    # InfoNCE pulling FoodOn-connected ingredients together in the learned
    # representation: regularises with the structure of ALL 22,784 ingredient
    # nodes, not just the ~100 labeled ones.  Applied uniformly to every E2E
    # model so the comparison stays fair.  Set to 0 to disable.
    aux_structural_weight: float = (
        0.0  # 0 for vanilla baselines; proposed run overrides >0
    )
    aux_temperature: float = 0.07
    aux_neg: int = 5

    # --- optional warm-start of ingredient features from a pretrained
    #     unsupervised structural embedding (e.g. TransE on the same graph).
    #     This keeps the model end-to-end & transductive but seeds it with the
    #     full-graph structure that the two-stage pipeline relies on.
    pretrained_emb: str | None = None  # path to an embeddings .pkl (name -> vec)
    pretrained_trainable: bool = True  # fine-tune the seeded features end-to-end

    # --- protocol ---
    k_folds: int = 5
    seed: int = 42
    device: str = "cuda"
    amp: bool = True
