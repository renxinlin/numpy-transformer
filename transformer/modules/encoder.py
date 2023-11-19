try:
    import cupy as np
    is_cupy_available = True
except:
    import numpy as np
    is_cupy_available = False
    
from transformer.layers.base.embedding import Embedding
from transformer.layers.base.dropout import Dropout
from transformer.layers.combined.encoder_layer import EncoderLayer
from transformer.layers.combined.positional_encoding import PositionalEncoding




class Encoder:
    def __init__(self, src_vocab_size, heads_num, layers_num, d_model, d_ff, dropout, max_length = 5000, data_type = np.float32):
        """

        :param src_vocab_size: 语料库大小
        :param heads_num: 8头注意力
        :param layers_num: 3层编码器
        :param d_model: 层输出大小 256 对于多头注意力是所有头的总大小
        :param d_ff: 全连接层大小 512
        :param dropout: dropout比例 0.1
        :param max_length: 句子最大长度 5000
        :param data_type:
        """
        self.token_embedding    = Embedding(src_vocab_size, d_model, data_type)
        self.position_embedding = PositionalEncoding(max_length, d_model, dropout, data_type)

        self.layers = []
        for _ in range(layers_num):
            self.layers.append(EncoderLayer(d_model, heads_num, d_ff, dropout, data_type))

        self.dropout = Dropout(dropout, data_type)
        self.scale = np.sqrt(d_model).astype(data_type) 

    def forward(self, src, src_mask, training):
        """

        :param src: 32 * max sequence
        :param src_mask:
        :param training:
        :return:
        """
        # 此编码
        # 输出 batch * segment size * 语料库 => batch * segment size * 256
        src = self.token_embedding.forward(src) * self.scale
        # 位置编码  batch * segment size * 256
        src = self.position_embedding.forward(src)
        # dropout部分特征0.1
        src = self.dropout.forward(src, training)
        # 三个编码器
        for layer in self.layers:
            # 位置编码  batch * segment size * 256
            src = layer.forward(src, src_mask, training)

        return src

    def backward(self, error):
        
        for layer in reversed(self.layers):
            error = layer.backward(error)
        
        error = self.dropout.backward(error)
        error = self.position_embedding.backward(error) * self.scale
        error = self.token_embedding.backward(error)


    def set_optimizer(self, optimizer):
        self.token_embedding.set_optimizer(optimizer)

        for layer in self.layers:
            layer.set_optimizer(optimizer)

    def update_weights(self):
        layer_num = 1
        layer_num = self.token_embedding.update_weights(layer_num)

        for layer in self.layers:
            layer_num = layer.update_weights(layer_num)

