from transformer.layers.base.dropout import Dropout
from transformer.layers.combined.self_attention import MultiHeadAttention
from transformer.layers.combined.positionwise_feed_forward import PositionwiseFeedforward
from transformer.layers.base.layer_norm import LayerNormalization


class EncoderLayer:
    def __init__(self, d_model, heads_num, d_ff, dropout, data_type):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = LayerNormalization(d_model, epsilon=1e-6, data_type=data_type)
        self.ff_layer_norm       = LayerNormalization(d_model, epsilon=1e-6, data_type=data_type)
        self.self_attention = MultiHeadAttention(d_model, heads_num, dropout, data_type)
        self.position_wise_feed_forward = PositionwiseFeedforward(d_model, d_ff, dropout)

        self.dropout = Dropout(dropout, data_type)

    def forward(self, src, src_mask, training):
        # layer1   batch * segment size * 256 （词嵌入转换得到）
        # layer2   batch * segment size * 256 （词嵌入转换得到）
        # layer3   batch * segment size * 256 （词嵌入转换得到）
        _src, _ = self.self_attention.forward(src, src, src, src_mask, training) # Self-Attention 操作
        src = self.self_attention_norm.forward(src + self.dropout.forward(_src, training)) # Add & Norm 操作
        # 两个线性变换和一个非线性激活函数

        """
        Position-wise Feed-Forward是Transformer模型中Encoder和Decoder层都会使用的一种操作。它包括两个线性变换和一个非线性激活函数，通常是ReLU。这个操作可以被描述为：
        首先，对每个序列位置的特征进行独立的线性变换，将输入张量的每个位置的特征映射到一个更高维度的空间。
        然后，对得到的新的特征向量应用非线性激活函数，通常是ReLU。
        最后，再进行另一个线性变换，将维度缩减回原来的维度。
        这个操作的作用在于充分利用位置信息和特征之间的关系，以及提供多样化的特征表示。这有助于模型更好地理解输入序列中不同位置的信息，并提升模型的表达能力。
        总之，Position-wise Feed-Forward操作是Transformer模型中重要的组成部分，用于增强模型对序列数据的建模能力
        """
        _src = self.position_wise_feed_forward.forward(src, training) # Position-wise Feed-Forward 操作
        src = self.ff_layer_norm.forward(src + self.dropout.forward(_src, training)) # Add & Norm 操作

        return src

    def backward(self, error):
        error = self.ff_layer_norm.backward(error)

        _error = self.position_wise_feed_forward.backward(self.dropout.backward(error))
        error = self.self_attention_norm.backward(error + _error)
        
        _error, _error2, _error3 = self.self_attention.backward(self.dropout.backward(error))

        return _error +_error2 +_error3 + error

    def set_optimizer(self, optimizer):
        self.self_attention_norm.set_optimizer(optimizer)
        self.ff_layer_norm.set_optimizer(optimizer)
        self.self_attention.set_optimizer(optimizer)
        self.position_wise_feed_forward.set_optimizer(optimizer)

    def update_weights(self, layer_num):
        layer_num = self.self_attention_norm.update_weights(layer_num)
        layer_num = self.ff_layer_norm.update_weights(layer_num)
        layer_num = self.self_attention.update_weights(layer_num)
        layer_num = self.position_wise_feed_forward.update_weights(layer_num)

        return layer_num