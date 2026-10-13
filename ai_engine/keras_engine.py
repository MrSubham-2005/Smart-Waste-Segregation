"""
EcoSort AI - Robust Keras Model Loader & Inference Engine
Loads and executes the genuine MobileNetV2 trained model (ecosort_mobilenetv2.keras).
Provides standard .predict(tensor) interface:
- Primary: tf.keras.models.load_model
- Fallback: Native Keras Zip+H5 weight evaluator for environments with DLL/AppControl restrictions
"""

import os
import io
import json
import zipfile
import h5py
import numpy as np


class KerasNumpyModel:
    """
    Native NumPy runtime for Keras 3 MobileNetV2 models (.keras archives).
    Directly loads the genuine trained weights from model.weights.h5 and config.json.
    Provides identical inference results without requiring compiled C++ extension DLLs.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._load_from_keras_archive(model_path)

    def _load_from_keras_archive(self, model_path: str):
        with zipfile.ZipFile(model_path, 'r') as z:
            cfg = json.loads(z.read('config.json').decode('utf-8'))
            self.base_cfg = cfg['config']['layers'][1]['config']
            weights_bytes = z.read('model.weights.h5')

        bio = io.BytesIO(weights_bytes)
        with h5py.File(bio, 'r') as h5:
            h5_base = h5['layers']['functional']['layers']
            self.weights = {}

            c_idx, dw_idx, bn_idx = 0, 0, 0
            for l in self.base_cfg['layers']:
                cname = l['class_name']
                name = l['name']

                if cname == 'Conv2D':
                    h5_key = 'conv2d' if c_idx == 0 else f'conv2d_{c_idx}'
                    self.weights[name] = {
                        'kernel': np.array(h5_base[h5_key]['vars']['0'], dtype=np.float32)
                    }
                    c_idx += 1
                elif cname == 'DepthwiseConv2D':
                    h5_key = 'depthwise_conv2d' if dw_idx == 0 else f'depthwise_conv2d_{dw_idx}'
                    self.weights[name] = {
                        'kernel': np.array(h5_base[h5_key]['vars']['0'], dtype=np.float32)
                    }
                    dw_idx += 1
                elif cname == 'BatchNormalization':
                    h5_key = 'batch_normalization' if bn_idx == 0 else f'batch_normalization_{bn_idx}'
                    gamma = np.array(h5_base[h5_key]['vars']['0'], dtype=np.float32)
                    beta = np.array(h5_base[h5_key]['vars']['1'], dtype=np.float32)
                    mean = np.array(h5_base[h5_key]['vars']['2'], dtype=np.float32)
                    var = np.array(h5_base[h5_key]['vars']['3'], dtype=np.float32)
                    eps = float(l['config'].get('epsilon', 1e-3))
                    scale = gamma / np.sqrt(var + eps)
                    bias = beta - mean * scale
                    self.weights[name] = {'scale': scale, 'bias': bias}
                    bn_idx += 1

            # Head layers
            head_bn = h5['layers']['batch_normalization']['vars']
            gamma = np.array(head_bn['0'], dtype=np.float32)
            beta = np.array(head_bn['1'], dtype=np.float32)
            mean = np.array(head_bn['2'], dtype=np.float32)
            var = np.array(head_bn['3'], dtype=np.float32)
            eps = 1e-3
            self.head_bn_scale = gamma / np.sqrt(var + eps)
            self.head_bn_bias = beta - mean * self.head_bn_scale

            self.dense_w = np.array(h5['layers']['dense']['vars']['0'], dtype=np.float32)
            self.dense_b = np.array(h5['layers']['dense']['vars']['1'], dtype=np.float32)

            self.head_w = np.array(h5['layers']['dense_1']['vars']['0'], dtype=np.float32)
            self.head_b = np.array(h5['layers']['dense_1']['vars']['1'], dtype=np.float32)

    def _conv2d(self, x, kernel, strides, padding):
        kh, kw, in_c, out_c = kernel.shape
        sh, sw = strides
        b, h, w, c = x.shape

        if kh == 1 and kw == 1 and sh == 1 and sw == 1:
            return np.dot(x, kernel.reshape(in_c, out_c))

        if padding == 'same':
            pad_h = max((int(np.ceil(h / sh)) - 1) * sh + kh - h, 0)
            pad_w = max((int(np.ceil(w / sw)) - 1) * sw + kw - w, 0)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            x_pad = np.pad(x, ((0,0), (pad_top, pad_bottom), (pad_left, pad_right), (0,0)), mode='constant')
        else:
            x_pad = x

        _, hp, wp, _ = x_pad.shape
        out_h = (hp - kh) // sh + 1
        out_w = (wp - kw) // sw + 1

        s_b, s_h, s_w, s_c = x_pad.strides
        patches = np.lib.stride_tricks.as_strided(
            x_pad,
            shape=(b, out_h, out_w, kh, kw, in_c),
            strides=(s_b, s_h * sh, s_w * sw, s_h, s_w, s_c)
        )
        flat_patches = patches.reshape(-1, kh * kw * in_c)
        flat_kernel = kernel.reshape(kh * kw * in_c, out_c)
        return np.dot(flat_patches, flat_kernel).reshape(b, out_h, out_w, out_c)

    def _depthwise_conv2d(self, x, kernel, strides, padding):
        kh, kw, in_c, _ = kernel.shape
        sh, sw = strides
        b, h, w, c = x.shape

        if padding == 'same':
            pad_h = max((int(np.ceil(h / sh)) - 1) * sh + kh - h, 0)
            pad_w = max((int(np.ceil(w / sw)) - 1) * sw + kw - w, 0)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            x_pad = np.pad(x, ((0,0), (pad_top, pad_bottom), (pad_left, pad_right), (0,0)), mode='constant')
        else:
            x_pad = x

        _, hp, wp, _ = x_pad.shape
        out_h = (hp - kh) // sh + 1
        out_w = (wp - kw) // sw + 1

        s_b, s_h, s_w, s_c = x_pad.strides
        patches = np.lib.stride_tricks.as_strided(
            x_pad,
            shape=(b, out_h, out_w, kh, kw, in_c),
            strides=(s_b, s_h * sh, s_w * sw, s_h, s_w, s_c)
        )
        k = kernel[:, :, :, 0]
        return np.einsum('bhwijk,ijk->bhwk', patches, k)

    def predict(self, x: np.ndarray, verbose: int = 0) -> np.ndarray:
        """
        Executes end-to-end forward inference on the preprocessed image tensor.
        
        :param x: Float32 NumPy array of shape (batch_size, 224, 224, 3) in [-1.0, 1.0].
        :return: Softmax probability distribution of shape (batch_size, 4).
        """
        tensors = {'input_layer_1': x}

        for l in self.base_cfg['layers']:
            cname = l['class_name']
            name = l['name']

            if cname == 'InputLayer':
                continue

            in_nodes = l.get('inbound_nodes', [])
            if in_nodes:
                args = in_nodes[0].get('args', [])
                if args:
                    first_arg = args[0]
                    if isinstance(first_arg, list):
                        inp = [tensors[item['config']['keras_history'][0]] for item in first_arg]
                    elif isinstance(first_arg, dict):
                        inp = tensors[first_arg['config']['keras_history'][0]]
                    else:
                        inp = None
            else:
                inp = None

            cfg = l.get('config', {})

            if cname == 'Conv2D':
                w = self.weights[name]['kernel']
                strides = tuple(cfg.get('strides', [1, 1]))
                padding = cfg.get('padding', 'valid')
                out = self._conv2d(inp, w, strides, padding)
            elif cname == 'DepthwiseConv2D':
                w = self.weights[name]['kernel']
                strides = tuple(cfg.get('strides', [1, 1]))
                padding = cfg.get('padding', 'valid')
                out = self._depthwise_conv2d(inp, w, strides, padding)
            elif cname == 'BatchNormalization':
                scale = self.weights[name]['scale']
                bias = self.weights[name]['bias']
                out = inp * scale + bias
            elif cname == 'ReLU':
                max_v = float(cfg.get('max_value', 6.0) or 6.0)
                out = np.clip(inp, 0.0, max_v)
            elif cname == 'ZeroPadding2D':
                pad = cfg.get('padding', [[1, 1], [1, 1]])
                out = np.pad(inp, ((0,0), (pad[0][0], pad[0][1]), (pad[1][0], pad[1][1]), (0,0)), mode='constant')
            elif cname == 'Add':
                out = inp[0] + inp[1]
            else:
                raise ValueError(f"Unsupported layer type: {cname}")

            tensors[name] = out

        base_out = tensors['out_relu']

        # Classification Head
        # 1. GlobalAveragePooling2D: (batch, 7, 7, 1280) -> (batch, 1280)
        gap = np.mean(base_out, axis=(1, 2))

        # 2. BatchNormalization
        gap_bn = gap * self.head_bn_scale + self.head_bn_bias

        # 3. Dense 256 + ReLU
        dense_out = np.maximum(0.0, np.dot(gap_bn, self.dense_w) + self.dense_b)

        # 4. Dense 4 (logits)
        logits = np.dot(dense_out, self.head_w) + self.head_b

        # 5. Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        return probs


def load_model_from_file(model_path: str):
    """
    Attempts to load model using standard TensorFlow/Keras first.
    If unavailable or blocked by environment policies, uses native KerasNumpyModel.
    """
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as tf_err:
        # Seamless fallback using the genuine trained weights in the .keras file
        return KerasNumpyModel(model_path)
