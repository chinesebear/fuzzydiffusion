import torch

class Options:
    def __init__(self, name) -> None:
        self.name= name
    def name(self):
        return self.name

# project gloal parameter
options = Options("Model")
options.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
options.evaluate_path="/home/yang/sda/github/evaluate/"
options.base_path="/home/yang/sda/github/fuzzydiffusion/"
options.model_parameter_path = options.base_path+"output/pth/"
options.img_path = options.base_path+"output/img/"
options.seed_id = 10
options.SOS = 0 # start of sentence
options.EOS = 1 # End of sentence
options.PAD = 2 # padding token
options.UNK = 3 # unknown token, word frequency low
options.count_max = 2000
options.size_max = 50
options.seq_max = 2000
options.epoch= 5
options.feature_num = 2 # [len, HF]
options.rule_num =3  ## fuzzys2s rule num
options.label_num = 2048
options.cluster_num = options.rule_num
options.iter_num = 150
options.h = 10.0
options.sen_len_max = 1024
options.high_freq_limit = 100
options.drop_out = 0.1
options.learning_rate = 0.0001
options.T = 1000
options.batch_size = 8
options.img_width = 128
options.img_hight = 128
options.img_size = (options.img_width,options.img_hight)

unet = Options("UNet")
unet.channel = 128
unet.channel_mult = [1, 2, 2, 2]
unet.attn = [1]
unet.num_res_blocks = 2
unet.dropout = 0.1
options.unet = unet

diff = Options("diffusion")
diff.beta_1 = 1e-4
diff.beta_T = 0.02
options.diff = diff

def setting_info():
    output = ""
    output += "rule_num: "+ str(options.rule_num)+", "
    output += "embedding: "+ str(trans.embedding_dim)+", "
    output += "h: "+ str(options.h)+", "
    output += "drop_out: "+ str(options.drop_out)+", "
    output += "learning_rate: "+ str(options.learning_rate)
    return output