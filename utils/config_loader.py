import yaml

def load_config(path:str):
    """
    load YAML config file from given path 
    
    args:
        path(str): path to the YAML file.
        
    returns:
        dict: parsed yaml contests as a dictionary 
    """
    with open(path,'r') as file:
        return yaml.safe_load(file)