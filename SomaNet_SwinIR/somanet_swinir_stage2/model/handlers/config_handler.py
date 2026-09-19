import yaml
import os
from typing import Union, Dict, Any
import json
from datetime import datetime

class ConfigHandler:
    DEFAULT_CHECKPOINT_DIR = './output/checkpoints'
    DEFAULT_LOG_DIR = './output/logs'
    DEFAULT_INFERENCE_RESULTS_DIR = './inference_results/inference_results'
    DEFAULT_INFERENCE_LOG_DIR = './inference_results/logs'
    
    def __init__(self, config: Union[str, Dict[str, Any]]):
        ConfigHandler._print_now_time()
        self.config = self._load_config(config)
        self._ensure_directories()
        self._print_and_save_config()

    @staticmethod
    def _print_now_time():
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        print("Current Date and Time:", current_time)
        
    def _load_config(self, config: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(config, str):
            if not os.path.isfile(config):
                raise FileNotFoundError(f"The configuration file '{config}' does not exist.")
            with open(config, 'r') as file:
                config = yaml.safe_load(file)
        elif isinstance(config, dict):
            pass
        else:
            raise ValueError("The config parameter must be a file path or a dictionary.")
        return config

    def _create_directory(self, path: str, description: str):
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"{description} directory created at {path}")
        else:
            print(f"{description} directory already exists at {path}")

    def _ensure_directories(self):
        mode = self.config.get('mode')
        if mode == 'train':
            self._create_directory(self.config.get('checkpoint_dir', self.DEFAULT_CHECKPOINT_DIR), "Checkpoint")
            self._create_directory(self.config.get('log_dir', self.DEFAULT_LOG_DIR), "Log")
            print("\n")
        elif mode == 'inference':
            self._create_directory(self.config.get('inference_results_dir', self.DEFAULT_INFERENCE_RESULTS_DIR), "Inference results")
            self._create_directory(self.config.get('log_dir', self.DEFAULT_INFERENCE_LOG_DIR), "Log")
            print("\n")
        else:
            raise ValueError("The mode can either be 'train' or 'inference'.")

    def _convert_sets_to_lists(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """ Convert all sets in the configuration to lists """
        for key, value in config.items():
            if isinstance(value, set):
                config[key] = list(value)
            elif isinstance(value, dict):
                config[key] = self._convert_sets_to_lists(value)
        return config

    def _print_and_save_config(self):
        config_to_print = self._convert_sets_to_lists(self.config.copy())
        
        if config_to_print.get('print_config', False):
            # Print config to console in a beautiful format
            print("\nConfiguration:")
            print(json.dumps(config_to_print, indent=4))
            print("\n")

        # Save config to log directory
        config_save_path = os.path.join(self.config.get('log_dir', self.DEFAULT_LOG_DIR), 'config.yaml')
        with open(config_save_path, 'w') as file:
            yaml.dump(config_to_print, file, default_flow_style=False, sort_keys=False, indent=4)

