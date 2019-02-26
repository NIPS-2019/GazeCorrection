import os


class Config:

    CUDA = True
    use_sp = True
    shuffle = True
    is_training = True

    step = 0
    lam_fp = 1
    gpu_id = 0 
    lr_decay = 1
    beta1 = 0.5
    beta2 = 0.999
    loss_type = 1
    pos_number = 4
    lam_recon = 10
    batch_size = 16
    batch_num = 200
    capacity = 5000
    num_threads = 10
    image_size = 256
    max_iters = 500000
    test_step = 100000
    g_learning_rate = 0.0001
    d_learning_rate = 0.0001
    hwc = [image_size, image_size, 3]

    pretrain_model_dirname = "read_model"
    attr_0_filename = "eye_position_0.txt"
    attr_1_filename= "eye_position_1.txt"
    data_dirname = "eye_ijcai"

    model_dirname = "models"
    result_dirname = "results"
    sample_dirname = "samples"
    log_dirname = "logs"

    @property
    def base_path(self):
        return os.path.abspath(os.curdir)


    @property
    def dataset_dir(self):
        dataset_dir = os.path.join(self.base_path,self.data_dirname)
        if not os.path.exists(dataset_dir):
            raise ValueError('Please specify a data dir.')
        return dataset_dir

    @property
    def exp_dir(self):
        exp_dir = self.base_path

        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        return exp_dir

    @property
    def pretrain_model_dir(self):
        model_path = os.path.join(self.exp_dir, self.pretrain_model_dirname)

        if not os.path.exists(model_path):
            os.makedirs(model_path)
        return model_path

    @property
    def model_dir(self):
        model_path = os.path.join(self.exp_dir, self.model_dirname)

        if not os.path.exists(model_path):
            os.makedirs(model_path)
        return model_path

    @property
    def log_dir(self):
        log_dir = os.path.join(self.exp_dir, self.log_dirname)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    @property
    def sample_dir(self):
        sample_path = os.path.join(self.exp_dir, self.sample_dirname)

        if not os.path.exists(sample_path):
            os.makedirs(sample_path)
        return sample_path

    @property
    def result_dir(self):
        sample_path = os.path.join(self.exp_dir, self.result_dirname)

        if not os.path.exists(sample_path):
            os.makedirs(sample_path)
        return sample_path





