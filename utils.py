import json

def load_model(model_name='Single Layer GELU'):
    with open("registered_models.json", "r") as f:
        data = json.load(f)
        ckpt = data[model_name]['checkpoint']
        ckpt_dir = '/'.join(ckpt.split('/')[:-1])
        return ckpt, ckpt_dir