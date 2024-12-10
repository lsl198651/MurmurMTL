# -*- coding: utf-8 -*-
import sys
from pathlib import Path

from model.ModelMTL import MurmurMTL

sys.dont_write_bytecode = True
PROJ_PATH = Path(__file__).parent.parent.as_posix()
sys.path.append(PROJ_PATH)
print(sys.path)

from configs.config import Config

from config import Config

VAR_DICT = {
    # "dataset_name": "UserBehavior",
    # "dataset_name": "ijcai15",
    "dataset_name": "KuaiRand",
    # "dataset_name": "QB-Video",
    # "dataset_name": "Ali-CCP",
    "num_workers": 8,
    "earlystop_patience": 3,
    "device_ids": 0,
    "batch_size": 2048,

    "epoch": 15,
    "warmup_epochs": 2,
    "full_arch_train_epochs": 3,
    "fine_tune_epochs": 5,
    "discretize_ops": 10,
    "test_epoch": 2,

    "model": {
        "name": "SuperNet",
        "kwargs": {
            "dropout": 0.2,
            "tower_layers": [64],
            # search space
            "n_expert_layers": 2,
            "n_experts": 4,
            "expert_module": {
                "in_features": 64,
                "out_features": 64,
                "n_layers": 3,
                "ops": [
                    "Identity", "MLP-16", "MLP-32", "MLP-64",
                    "MLP-128", "MLP-256", "MLP-512", "MLP-1024",
                ]
            }
        }
    }
}

DATASET_COLLECTION = {
    "ijcai15": {
        "dataset": "ijcai15",
        "dataset_path": "/data/datasets/ijcai15/",  # data/datasets/ijcai15
        "dataset_ext": "csv",
        "dense_fields": [],
        "sparse_fields": ["user_id:token", "item_id:token", "cat_id:token", "seller_id:token", "brand_id:token",
                          "age_range:token", "gender:token"],
        "label_fields": ["purchase:label", "favourite:label"],
        "task_types": ["classification", "classification"],
        "embedding_dim": 16,
        "criterions": ["bce", "bce"],
        "val_metrics": ["auc", "auc"],
    }}


def main():
    config = Config(r"D:\Shilong\new_murmur\01_code\AutoMTL\configs\default_nas.yaml", VAR_DICT).get_config_dict()
    if "dataset_name" in config:    # config dataset via console args 'dataset_name'
        config["dataset"]       = DATASET_COLLECTION[config["dataset_name"]]["dataset"]
        config["dataset_path"]  = DATASET_COLLECTION[config["dataset_name"]]["dataset_path"]
        config["dataset_ext"]   = DATASET_COLLECTION[config["dataset_name"]]["dataset_ext"]
        config["dense_fields"]  = DATASET_COLLECTION[config["dataset_name"]]["dense_fields"]
        config["sparse_fields"] = DATASET_COLLECTION[config["dataset_name"]]["sparse_fields"]
        config["label_fields"]  = DATASET_COLLECTION[config["dataset_name"]]["label_fields"]
        config["task_types"]    = DATASET_COLLECTION[config["dataset_name"]]["task_types"]
        config["embedding_dim"] = DATASET_COLLECTION[config["dataset_name"]]["embedding_dim"]
        config["criterions"]    = DATASET_COLLECTION[config["dataset_name"]]["criterions"]
        config["val_metrics"]   = DATASET_COLLECTION[config["dataset_name"]]["val_metrics"]

    searcher = MurmurMTL(config, rank=0)
    searcher.fit()
    auc = searcher.evaluate()
    print(f"Final auc.mean: {auc.mean()}")


if __name__ == "__main__":
    main()
