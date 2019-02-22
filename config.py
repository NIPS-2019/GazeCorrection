import os
import time

class Config:

    CUDA = True
    use_sp = True
    shuffle = True
    is_training = True

    step = 0
    lam_fp = 1
    gpu_id = 0
    beta1 = 0.5
    lr_decay = 1
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

    cuda_number = "0"
    logs_name = "logs"
    dir_path = "NewGazeData"
    read_model_name = "read_model"
    write_model_name = "write_model"
    test_name = "test_img_inpainting"
    sample_name = "sample_img_inpainting"
    dataset_txt0 = "/eye_position_new_gaze_0.txt"
    dataset_txt1 = "/eye_position_new_gaze_1.txt"
    data_dir_path = "/eye_ijcai/"

    @property
    def base_path(self):
        return os.path.abspath(os.curdir)

    @property
    def data_dir(self):
        data_dir = os.path.join(self.base_path,self.data_dir_path )

        if not os.path.exists(data_dir):
            raise ValueError('Please specify a data dir.')
        return data_dir

    @property
    def exp_dir(self):
        exp_dir = self.base_path

        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        return exp_dir

    @property
    def read_model_path(self):
        model_path = os.path.join(self.exp_dir, self.read_model_name)

        if not os.path.exists(model_path):
            os.makedirs(model_path)
        return model_path

    @property
    def write_model_path(self):
        model_path = os.path.join(self.exp_dir, self.write_model_name)

        if not os.path.exists(model_path):
            os.makedirs(model_path)
        return model_path

    @property
    def log_dir(self):
        log_dir = os.path.join(self.exp_dir, self.logs_name)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    @property
    def sample_path(self):
        sample_path = os.path.join(self.exp_dir, self.sample_name)

        if not os.path.exists(sample_path):
            os.makedirs(sample_path)
        return sample_path

    @property
    def test_sample_path(self):
        sample_path = os.path.join(self.exp_dir, self.test_name)

        if not os.path.exists(sample_path):
            os.makedirs(sample_path)
        return sample_path

    def get_iter_num(self):
        return int(os.listdir(self.read_model_path)[0].split(".")[0].split("_")[1])


    operation_name = time.strftime("%Y-%m-%d",time.localtime(time.time()))


