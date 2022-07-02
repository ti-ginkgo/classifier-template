from omegaconf import OmegaConf


def load_config(config_name):
    default_config = OmegaConf.load("./configs/default_config.yaml")
    config = OmegaConf.load(f"./configs/{config_name}")
    config = OmegaConf.merge(default_config, config)

    return config
