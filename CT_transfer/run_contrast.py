import yaml
import os

from datasets.data_pretrain_ginet import CrystalDatasetWrapper
from graph_contrast import GraphContrastive

import warnings
warnings.simplefilter("ignore")
warnings.warn("deprecated", UserWarning)
warnings.warn("deprecated", FutureWarning)


def main():
    config = yaml.load(open("config_contrast.yaml", "r"), Loader=yaml.FullLoader)
    print(config)
    dataset = CrystalDatasetWrapper(config['batch_size'], **config['dataset'])

    graph_contrast = GraphContrastive(dataset, config)
    graph_contrast.train()


if __name__ == "__main__":
    main()
