import pandas as pd
import json
import os
import magic
import fitz
import yaml

def load_csv(path):
    return pd.read_csv(path)

def save_csv(df,path):
    return df.to_csv(path,index=False)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)
    
def save_json(data,path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def load_yaml(path):
    with open(path,'r') as f:
        return yaml.safe_load(f)
    
def load_text_file(path):
    with open(path,'r',encoding='utf-8') as f:
        return f.read()
def extract_text_from_pdf(path):
    doc=fitz.open(path)
    return '\n'.join([page.get_text() for page in doc])

def get_file_type(path):
    mime = magic.Magic(mime=True)
    return mime.from_file(path)