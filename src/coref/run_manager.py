"""
Supply output_dir or not. This determines the mode of operation.

default mode:
If output_dir is not supplied, then it is implicitly constructed using 
the config file path, output_root, and experiments_root.

pipeline mode:
If output_dir is supplied, then we have the option to supply either a config file
or the config as named arguments. An attempt will be made to coerce types to match
the config types.

The flags output_dir, output_root, and experiments_root can be set in config file 
as _output_dir, _output_root, and _experiments_root respectively.

However, command line flags will always take precedence over config file values.

"""
import os
import shutil
import yaml
import pathlib

CONFIG_META_KEYS = ['_output_dir', '_output_root', '_experiments_root']

def load_cfg(fp):
    with open(fp) as f:
        cfg = yaml.full_load(f)
    
    meta_kwargs = {k: cfg.pop(k) for k in CONFIG_META_KEYS if k in cfg}
    return cfg, meta_kwargs


def int_or_none(fn):
    try:
        return int(fn)
    except:
        return None


def get_family_name(config_path, runs_root, experiments_root):
    rel_path = os.path.relpath(config_path, experiments_root)
    family_name, _ = os.path.splitext(rel_path)
    return family_name


def get_run_dir_parent(config_path, runs_root, experiments_root):
    rel_path = os.path.relpath(config_path, experiments_root)
    family_name, _ = os.path.splitext(rel_path)
    run_dir = os.path.join(runs_root, family_name)
    return run_dir


def build_run_dir(config_path, runs_root, experiments_root):
    run_dir = get_run_dir_parent(config_path, runs_root, experiments_root)
    if not os.path.exists(run_dir):
        highest_run = 0
    else:
        dirs = [
            d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d))
        ]
        ints = [int_or_none(d) for d in dirs]
        highest_run = max([x for x in ints if x is not None]) + 1
    run_dir = os.path.join(run_dir, str(highest_run))
    return run_dir


import logging


def get_run_dir(config_path, runs_root, experiments_root):
    experiments_root = os.path.abspath(experiments_root)
    config_path = os.path.abspath(config_path)
    runs_root = os.path.abspath(runs_root)
    assert os.path.samefile(
        experiments_root, os.path.commonpath([experiments_root, config_path])
    )
    run_dir = build_run_dir(config_path, runs_root, experiments_root)
    return pathlib.Path(run_dir)


import dataclasses
import typeguard


class Cfg:
    def __post_init__(self):
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            try:
                typeguard.check_type(value, field.type)
            except typeguard.TypeCheckError as e:
                if dataclasses.is_dataclass(field.type):
                    new_value = field.type(**value)
                    setattr(self, field.name, new_value)
                elif isinstance(value, str):
                    if any(field.type is t for t in [int, float]):
                        try:
                            new_value = field.type(value)
                        except:
                            raise TypeError(
                                f"Failed to coerce {value} to {field.type}"
                            )
                    elif field.type is bool:
                        if value.lower() in ["true", "t", "1"]:
                            new_value = True
                        elif value.lower() in ["false", "f", "0"]:
                            new_value = False
                        else:
                            raise TypeError(
                                f"Failed to coerce {value} to {field.type}"
                            )
                    else:
                        # last ditch attempt
                        try:
                            new_value = eval(value)
                        except:
                            raise TypeError(
                                f"Failed to coerce {value} to {field.type}"
                            )
                    logging.warning(f'Coercing {field.name} from {value}: {type(value)} to {field.type}: {new_value}')
                    setattr(self, field.name, new_value)
            value = getattr(self, field.name)
            try:
                typeguard.check_type(value, field.type)
            except typeguard.TypeCheckError as e:
                logging.error(f"Typecheck error failed: {value} is not {field.type}")
                raise TypeError(
                    f"Typecheck error failed: {value} is not {field.type}"
                ) from e

    def save(self, path, check=True, meta_kwargs={}):
        assert all(k in CONFIG_META_KEYS for k in meta_kwargs)
        meta_kwargs = {k: str(v) for k, v in meta_kwargs.items()} # convert path to str
        s = yaml.dump({**dataclasses.asdict(self), **meta_kwargs}, sort_keys=False)
        if not os.path.exists(os.path.dirname(path)):
            print(f"making cfg path {os.path.dirname(path)}")
            os.makedirs(os.path.dirname(path))
        with open(path, "w") as f:
            f.write(s)
        if check:
            loaded_d, loaded_meta_kwargs = load_cfg(path)
            other = self.__class__(**loaded_d)
            assert self == other and meta_kwargs == loaded_meta_kwargs

def args_to_dict(args):
    args_dict = {}
    args = sum([arg.split("=") for arg in args], [])
    is_flags = [arg.startswith("--") for arg in args]
    for i, (arg, is_flag) in enumerate(zip(args, is_flags)):
        if is_flag:
            if i + 1 < len(args) and not is_flags[i + 1]:
                args_dict[arg[2:]] = args[i + 1]
            else:
                args_dict[arg[2:]] = True
        else:
            assert i > 0 and is_flags[i - 1], f"Non-flag argument {arg} must follow a flag"
    
    return args_dict

def automain(main):
    # concept is inspired by sacred
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--experiments_root", type=str, default=None)
    default_expt_paths = {
        'output_root': 'runs',
        'experiments_root': 'experiments'
    }

    def wrapped_main(*, mode='default', config, output_root=None, experiments_root=None, config_path=None, output_dir=None, dry_run=False):
        '''
        Args:
            mode: Literal['default', 'pipeline']
            dry_run: bool - If True, return cfg and output_dir without running main
        
        In default mode:
        Args:
            config: str - Path to config file
            output_root: str - Path to output root
            experiments_root: str - Path to experiments root
        
        In pipeline mode:
        Args:
            config_dict: dict - Config dictionary
            OR
            config: str - Path to config file
            output_dir: str - Path to output directory
        '''
        if mode == 'default':
            output_dir = get_run_dir(
                config_path, output_root, experiments_root,
            )
        if mode == 'pipeline':
            output_dir = pathlib.Path(output_dir)
        if dry_run:
            return config, output_dir
        else:
            os.makedirs(output_dir, exist_ok=True)
            logging.basicConfig(
                level=logging.INFO,
                handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler(os.path.join(output_dir, "logs.txt")),
                ],
            )
            with open(output_dir / "config.yaml", "w") as f:
                yaml.dump(config, f, sort_keys=False)
            logging.info(f"Starting run on {config}")
            logging.info(f"Output to {output_dir}")
            res = main(config, output_dir)
            logging.info("Completed")
            with open(output_dir / "done.out", "w") as f:
                f.write("done\n")
            return (res, config, output_dir)

    if main.__module__ == "__main__":
        args, other_args = parser.parse_known_args()
        expt_paths = {
            'output_dir': args.output_dir,
            'output_root': args.output_root,
            'experiments_root': args.experiments_root
        }

        if args.config is None:
            config = args_to_dict(other_args)
            config_path = None
        else:
            config, meta_kwargs = load_cfg(args.config)
            config_path = args.config
            args_config = args_to_dict(other_args)
            for key, value in args_config.items():
                if key in config:
                    print(f'Warning: Overriding config value {key}: {config.get(key)} with {value}')
                config[key] = value
            for meta_key, meta_value in meta_kwargs.items():
                if expt_paths[meta_key[1:]] is None:
                    expt_paths[meta_key[1:]] = meta_value
                else:
                    print(f'Warning: Overriding config value {meta_key}: {meta_value} with {expt_paths[meta_key[1:]]}')

        for key in default_expt_paths:
            if expt_paths[key] is None:
                expt_paths[key] = default_expt_paths[key]
        

        if expt_paths['output_dir'] is None:
            wrapped_main(
                mode='default',
                config=config,
                config_path=config_path,
                output_root=expt_paths['output_root'],
                experiments_root=expt_paths['experiments_root'],
            )
        else:
            wrapped_main(
                mode='pipeline',
                config=config,
                config_path=config_path,
                output_dir=expt_paths['output_dir'],
            )

    return wrapped_main