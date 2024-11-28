import sys
import os
from pathlib import Path
import nni


sys.dont_write_bytecode = True
PROJ_PATH = Path(__file__).parent.parent.as_posix()
sys.path.append(PROJ_PATH)
print(sys.path)

from config import Config


def main():

    config = Config(r"D:\Shilong\new_murmur\01_code\AutoMTL\configs\default_nas.yaml", VAR_DICT).get_config_dict()

    searcher = MurmurMTL(config, rank=0)
    searcher.fit()
    auc = searcher.evaluate()
    print(f"Final auc.mean: {auc.mean()}")



if __name__ == "__main__":
    main()
