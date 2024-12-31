#!/usr/bin python3

import torch
from sources.inference import inference
from sources.config_loader import Config
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.post("/infer")
def infer(input_data: dict):
    config = Config()
    config.load_config('gan_config.json')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
    try:
        inference(device, config, input_data["output_file"], prod=True)
    except Exception as e:
        return {"output_path": "", "error": str(e)}
    return {"output_path": input_data["output_file"], "error": ""}

if __name__ == '__main__':
    uvicorn.run(debug=True, port=5050, host="0.0.0.0")