import os
from Dataset import Dataset
from config import Config
from GazeGAN import Inpainting_GAN



if __name__ == "__main__":
    config = Config()

    if config.CUDA:
        os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_number

    d_ob = Dataset(config)
    igan = Inpainting_GAN(d_ob, config)
    igan.build_model()

    if config.is_training:
        igan.train()
    else:
        igan.test()