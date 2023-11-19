try:
    import cupy as np
    is_cupy_available = True
except:
    import numpy as np
    is_cupy_available = False
    
from transformer.layers.base.dense import Dense
from transformer.layers.base.embedding import Embedding
from transformer.layers.base.dropout import Dropout
from transformer.layers.combined.decoder_layer import DecoderLayer
from transformer.activations import Identity, Softmax
from transformer.layers.combined.positional_encoding import PositionalEncoding




class Decoder:
    def __init__(self, trg_vocab_size, heads_num, layers_num, d_model, d_ff, dropout, max_length = 5000, data_type = np.float32):

        self.token_embedding    = Embedding(trg_vocab_size, d_model, data_type)
        self.position_embedding = PositionalEncoding(max_length, d_model, dropout, data_type)

        self.layers = []
        for _ in range(layers_num):
            self.layers.append(DecoderLayer(d_model, heads_num, d_ff, dropout, data_type))

        self.fc_out = Dense(inputs_num = d_model, units_num = trg_vocab_size, data_type = data_type)
        self.dropout = Dropout(dropout, data_type)
        self.scale = np.sqrt(d_model).astype(data_type)
        self.activation = Identity()#Activation(Identity())


    def forward(self, trg, trg_mask, src, src_mask, training):
        # 缩放的词嵌入向量
        trg = self.token_embedding.forward(trg) * self.scale
        # 位置编码
        trg = self.position_embedding.forward(trg)
        # dropout 0.1的比例
        trg = self.dropout.forward(trg, training)
        # 3层 decoder[8头]
        for layer in self.layers:
            trg, attention = layer.forward(trg, trg_mask, src, src_mask, training)
        # 线性输出 激活并分类 分类为语料表大小
        output = self.fc_out.forward(trg)
        # desc: 没有用softmax激活,也可以理解 直接取最大值进行分类
        activated_output = self.activation.forward(output)
        softmax = Softmax()
        softmax_active = softmax.forward(output)
        # 返回预测概率，以及最后一层的多头注意力权重
        return activated_output, attention

    def backward(self, error):
        error = self.activation.backward(error)
        
        error = self.fc_out.backward(error)
        
        self.encoder_error = 0
        for layer in reversed(self.layers):
            error, ecn_error = layer.backward(error)
            self.encoder_error += ecn_error


        error = self.dropout.backward(error)

        error = self.position_embedding.backward(error) * self.scale
        error = self.token_embedding.backward(error)

    def set_optimizer(self, optimizer):
        self.token_embedding.set_optimizer(optimizer)

        for layer in self.layers:
            layer.set_optimizer(optimizer)

        self.fc_out.set_optimizer(optimizer)

    def update_weights(self):
        layer_num = 1
        self.token_embedding.update_weights(layer_num)

        for layer in self.layers:
            layer_num = layer.update_weights(layer_num)

        self.fc_out.update_weights(layer_num)
