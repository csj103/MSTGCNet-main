import os
import torch
from models import MSTGCNet


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'MSTGCNet': MSTGCNet,
        }
        if args.model not in self.model_dict:
            raise ValueError(f"Unsupported model in this repository: {args.model}")

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        raise NotImplementedError

    def vali(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError
