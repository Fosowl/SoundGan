# SoundGAN

This repository contains a Generative Adversarial Network (GAN) designed to generate sounds of a specific type. The project is built with a MLOps pipeline to streamline data collection, model retraining, and deployment.

Features

- Sound Generation with GAN: The model generates audio samples of a particular type (e.g., nature sounds, instrumental music).

- YouTube Data Scraping: Automatically scrape YouTube videos to build datasets for training on new sound types.

- Retraining Pipeline: Retrain the GAN using newly scraped and preprocessed data.

- Backend API: Serve generated audio through an API built with Go and a Python microservice for inference.


## Project structure

```
.
├── .github/workflows/           # CI/CD workflows
│   └── ml-pipeline.yml          # GitHub Actions workflow for the MLOps pipeline
├── backend/                     # Backend API code
├── data/                        # Data collection and preprocessing scripts
├── gan/                         # GAN model code and training utilities
│   ├── runs/                    # Training logs and run artifacts
│   ├── save/                    # Checkpoints or saved models
│   ├── sources/                 # GAN source code
│   │   ├── config_loader.py     # Configuration loading utility
│   │   ├── discriminator.py     # Discriminator model definition
│   │   ├── generator.py         # Generator model definition
│   │   ├── inference.py         # Inference code for GAN
│   │   ├── notify.py            # Notification utility 
│   │   ├── plotting.py          # Plotting utilities
│   │   └── training.py          # GAN training logic
│   ├── app.py                   # Python microservice entry point
│   ├── main.py                  # Main script for training
│   ├── gan_config.json          # Configuration file for GAN parameters
├── Dockerfile                   # Dockerfile for microservice
├── docker-compose.yaml          # Docker Compose file for orchestrating services
├── requirements.txt             # Python dependencies
├── .gitignore                   # Ignored files and folders
├── LICENSE                      # Project license
└── README.md                    # Project documentation
```

## Installation and Setup

### 1. Clone the Repository

```
bash
Copy code
git clone https://github.com/yourusername/sound-gan.git  
cd sound-gan  
```

### 2. Run the Pipeline Locally

```
pip install -r requirements.txt  
sudo apt install ffmpeg  
```

**Run the data harvester**

```
bash data/yt_harverser/data_pipeline.sh  
```

**Train the GAN**

```
python3 gan.py --training  
```

**Deployment using Docker Compose**

To deploy the backend (Go API and Python microservice):

```
docker-compose up --build  
```

### Contributing
Feel free to fork the repository and open a pull request for any enhancements or bug fixes!

Inspired by the desire to explore GANs and MLOps principles.
Built as a demo project to combine sound generation and automation.

### License
This project is licensed under the MIT License. See the LICENSE file for details.