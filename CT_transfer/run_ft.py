import yaml
import os

from dataset.data_ft_ginet import CrystalDatasetWrapper
from finetune import FineTune

# import warnings
# warnings.simplefilter("ignore")
# warnings.warn("deprecated", UserWarning)


def main(config):
    # config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    dataset = CrystalDatasetWrapper(config['batch_size'], **config['dataset'])

    fine_tune = FineTune(dataset, config)
    loss, metric = fine_tune.train()
    
    return loss, metric


if __name__ == "__main__":
    
    config = yaml.load(open("config_ft.yaml", "r"), Loader=yaml.FullLoader)
    print(config)

    if 'BG' in config['dataset']['data_dir']:
        config['dataset']['task'] = 'regression'
        task_name = 'BG'
    elif 'E_form' in config['dataset']['data_dir']:
        config['dataset']['task'] = 'regression'
        task_name = 'E_form'
    elif 'Is_Metal' in config['dataset']['data_dir']:
        config['dataset']['task'] = 'classification'
        task_name = 'Is_Metal'
    elif 'MP-formation-energy' in config['dataset']['data_dir']:
        config['dataset']['task'] = 'regression'
        task_name = 'MP_energy'
    elif 'perovskites' in config['dataset']['data_dir']:
        config['dataset']['task'] = 'regression'
        task_name = 'Perovskites'
    elif 'jdft2d' in config['dataset']['data_dir']:
        config['dataset']['task'] = 'regression'
        task_name = 'jdft2d'
    elif 'phonons' in config['dataset']['data_dir']:
        config['dataset']['task'] = 'regression'
        task_name = 'phonons'
    
    loss, metric = main(config)
    
    import pandas as pd
    ftf = config['fine_tune_from'].split('/')[-1]
    fn = '{}_{}.csv'.format(ftf, task_name)
    print(fn)
    df = pd.DataFrame([[loss, metric]])
    df.to_csv(
        os.path.join('experiments', fn),
        # 'experiments/{}_{}_ft.csv'.format(config['fine_tune_from'], task_name), 
        mode='a', index=False, header=False
    )