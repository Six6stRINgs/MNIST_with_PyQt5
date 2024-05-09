import os
import torch
from PyQt5.QtCore import QThread, pyqtSignal

from data import load_mnist
from model import train_model, test_model


class TrainAndTestThread(QThread):
    progress_signal = pyqtSignal(int)
    text_signal = pyqtSignal(str)
    loss_signal = pyqtSignal(float)

    def __init__(self, model_config: dict, parent=None):
        super().__init__(parent)
        self.model_config = model_config
        self.train_loader = None
        self.test_loader = None
        self.model = None
        self.model_config_str = None
        self.acc = None

    def run(self):
        # TODO: Load data and train model in this thread, avoid blocking the UI
        self.train_loader, self.test_loader = load_mnist(self.model_config['batch_size'])
        self.model, self.model_config_str = train_model(self.train_loader, self.model_config,
                                                        self.progress_signal, self.text_signal, self.loss_signal)
        device = self.model_config['device']
        self.acc = test_model(self.test_loader, self.model, device, self.progress_signal, self.text_signal)

        self.model_config_str += '_Acc_{:.2f}_'.format(self.acc)
        # save model as onnx
        model_root = 'model'
        onnx_path = os.path.join('model', self.model_config_str + '.onnx')
        pt_path = os.path.join('model', self.model_config_str + '.pt')
        if not os.path.exists(model_root):
            os.makedirs(model_root)

        if str(self.model) != 'ViT':
            self.model.eval()
            with torch.no_grad():
                dummy_input = torch.randn(1, 1, 28, 28).to(device)
                torch.onnx.export(self.model, dummy_input, onnx_path, opset_version=16)
                res = 'Model saved as {}'.format(onnx_path)
        else:
            # 'aten::_transformer_encoder_layer_fwd' to ONNX opset version x is not supported
            torch.save(self.model, pt_path)
            res = 'Model saved as {}'.format(pt_path)

        print(res)
        self.text_signal.emit(res)

    def get_res(self):
        return self.model, self.train_loader, self.test_loader, self.model_config_str, self.acc
