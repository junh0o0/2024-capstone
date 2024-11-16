import argparse
import os
from typing import Optional

def build_default_arg_parser() -> argparse.ArgumentParser:
    try:
        import configargparse

        parser = configargparse.ArgumentParser(
            config_file_parser_class=configargparse.YAMLConfigFileParser,
        )
        parser.add(
            "--config",
            type=str,
            is_config_file=True,
            help="config file to agregate options",
        )
    except ImportError:
        parser = argparse.ArgumentParser()

    # Name and seed
    parser.add_argument("--train_path",type=str,required=True)
    parser.add_argument("--max_len",type=int,default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--warmup_ratio",type=float,default=0.1)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--max_grad_norm",type=int,default=1)
    parser.add_argument("--log_interval",type=int, default=200)
    parser.add_argument("--learning_rate",type=float,default=5e-5)

    # Device 
    parser.add_argument(
        "--device",
        help="select device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
    )
    
    return parser