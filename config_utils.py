import json

def read_config(cfg_path):
    with open(cfg_path, "r") as f:
        cfg=json.loads(f.read())
        print(cfg)
        return cfg

def add_val_to_array(cfg, prop_name, prop_val):
    if prop_val not in cfg[prop_name]:
        cfg[prop_name].append(prop_val)
            
def write_config(cfg, cfg_path):
    with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=4)
            
def load_json(path_):
    with open(path_, "r") as f:
        val_=json.loads(f.read())
        return val_
