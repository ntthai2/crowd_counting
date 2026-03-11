"""
Minimal mmcv shim for Python 3.13 compatibility.
mmcv 1.6.0 relies on pkgutil.ImpImporter removed in Python 3.12.
Covers: Config, DictAction, mmcv.utils.get_logger, mmcv.runner.BaseModule.
"""
import argparse
import importlib.util
import logging
import os

from mmcv import runner  # noqa: F401  (makes mmcv.runner importable)
from mmcv import utils   # noqa: F401  (makes mmcv.utils importable)


class ConfigDict(dict):
    """Dict subclass supporting dot-access to nested keys."""

    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError:
            raise AttributeError(f"'ConfigDict' has no attribute '{name}'")
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            value = ConfigDict(value)
            self[name] = value
        return value

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'ConfigDict' has no attribute '{name}'")


def _dict_to_configdict(obj):
    if isinstance(obj, dict):
        return ConfigDict({k: _dict_to_configdict(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        converted = [_dict_to_configdict(item) for item in obj]
        return type(obj)(converted)
    return obj


class Config:
    """Minimal Config that loads a Python config file and provides dot-access."""

    def __init__(self, cfg_dict=None):
        object.__setattr__(self, '_cfg_dict', ConfigDict(cfg_dict or {}))

    @staticmethod
    def fromfile(filepath):
        filepath = os.path.abspath(filepath)
        spec = importlib.util.spec_from_file_location("_mmcv_cfg_module", filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cfg_dict = {
            k: getattr(module, k)
            for k in dir(module)
            if not k.startswith('_')
        }
        cfg_dict = _dict_to_configdict(cfg_dict)
        return Config(cfg_dict)

    def merge_from_dict(self, options):
        if not options:
            return
        for key, value in options.items():
            keys = key.split('.')
            d = self._cfg_dict
            for k in keys[:-1]:
                if k not in d:
                    d[k] = ConfigDict()
                d = d[k]
            d[keys[-1]] = value

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, '_cfg_dict'), name)

    def __setattr__(self, name, value):
        if name == '_cfg_dict':
            object.__setattr__(self, name, value)
        else:
            self._cfg_dict[name] = value

    def __repr__(self):
        return f'Config({dict(self._cfg_dict)})'


class DictAction(argparse.Action):
    """argparse action to parse key=value pairs into a dict."""

    @staticmethod
    def _parse_int_float_bool(val):
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ('true', 'false'):
            return val.lower() == 'true'
        return val

    @staticmethod
    def _parse_value(value):
        if value.startswith('[') and value.endswith(']'):
            inner = value[1:-1]
            return [DictAction._parse_int_float_bool(v.strip()) for v in inner.split(',') if v.strip()]
        if value.startswith('(') and value.endswith(')'):
            inner = value[1:-1]
            return tuple(DictAction._parse_int_float_bool(v.strip()) for v in inner.split(',') if v.strip())
        return DictAction._parse_int_float_bool(value)

    def __call__(self, parser, namespace, values, option_string=None):
        options = getattr(namespace, self.dest, None) or {}
        for kv in values:
            if '=' not in kv:
                raise argparse.ArgumentError(self, f"Expected key=value, got: {kv}")
            k, v = kv.split('=', 1)
            options[k.strip()] = self._parse_value(v.strip())
        setattr(namespace, self.dest, options)
