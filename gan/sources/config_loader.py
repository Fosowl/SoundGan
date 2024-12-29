import json

class Config:
    def __init__(self):
        self.saveroot = None
        self.dataroot = None
        self.num_epochs = None
        self.original_image_size = None
        self.image_size = None
        self.workers = None
        self.batch_size = None
        self.nc = None
        self.nz = None
        self.ngf = None
        self.ndf = None
        self.lr_G = None
        self.lr_D = None
        self.beta1 = None
        self.ngpu = None
        self.initial_noise_std = None
        self.noise_decay_rate = None

    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.saveroot = config.get('saveroot')
        self.dataroot = config.get('dataroot')
        self.num_epochs = config.get('num_epochs')
        self.original_image_size = config.get('original_image_size')
        self.image_size = config.get('image_size')
        self.workers = config.get('workers')
        self.batch_size = config.get('batch_size')
        self.nc = config.get('nc')
        self.nz = config.get('nz')
        self.ngf = config.get('ngf')
        self.ndf = config.get('ndf')
        self.lr_G = config.get('lr_G')
        self.lr_D = config.get('lr_D')
        self.beta1 = config.get('beta1')
        self.ngpu = config.get('ngpu')
        self.initial_noise_std = config.get('initial_noise_std')
        self.noise_decay_rate = config.get('noise_decay_rate')