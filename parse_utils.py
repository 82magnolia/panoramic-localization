from collections import namedtuple
import configparser
from ast import literal_eval
import os


def parse_ini(config_path: str):
    read_config = configparser.ConfigParser()
    read_config.read(config_path)
    config_attribs = []
    data_dict = {}
    for section in read_config.sections():
        for (key, value) in read_config.items(section):
            config_attribs.append(key)
            data_dict[key] = parse_value(value)
            if value == 'None':  # Account for None
                data_dict[key] = None

    Config = namedtuple('Config', config_attribs)
    cfg = Config(**data_dict)
    return cfg


def parse_value(value):
    if value.replace('.', '', 1).replace('+', '', 1).replace('-', '', 1).replace('e', '', 1).isdigit():
        # Exponential format and decimal format should be accounted for
        return literal_eval(value)
    elif value == 'True' or value == 'False':
        if value == 'True':
            return True
        else:
            return False
    elif value == 'None':
        return None
    elif ',' in value:  # Config contains lists
        is_number = any(char.isdigit() for char in value.split(',')[0])
        items_list = value.split(',')

        if '' in items_list:
            items_list.remove('')
        if is_number:
            return [literal_eval(val) for val in items_list]
        else:
            if '\"' in items_list[0] and '\'' in items_list[0]:
                return [literal_eval(val.strip()) for val in items_list]
            else:
                return [val.strip() for val in items_list]
    else:
        return value


def save_ini(config_path, log_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    save_path = os.path.join(log_path, 'config.ini')
    with open(save_path, 'w') as f:
        config.write(f)

