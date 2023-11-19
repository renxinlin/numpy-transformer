from transformer.layers.base.dropout import Dropout
from transformer.layers.combined.self_attention import MultiHeadAttention
from transformer.layers.combined.positionwise_feed_forward import PositionwiseFeedforward
from transformer.layers.base.layer_norm import LayerNormalization

class DecoderLayer():
    def __init__(self, d_model, heads_num, d_ff, dropout, data_type):
        super(DecoderLayer, self).__init__()

        self.self_attention_norm = LayerNormalization(d_model, epsilon=1e-6, data_type = data_type)
        self.enc_attn_layer_norm = LayerNormalization(d_model, epsilon=1e-6, data_type = data_type)
        self.ff_layer_norm       = LayerNormalization(d_model, epsilon=1e-6, data_type = data_type)
        self.self_attention    = MultiHeadAttention(d_model, heads_num, dropout, data_type)
        self.encoder_attention = MultiHeadAttention(d_model, heads_num, dropout, data_type)
        self.position_wise_feed_forward = PositionwiseFeedforward(d_model, d_ff, dropout)

        self.dropout = Dropout(dropout, data_type)

    def forward(self, trg, trg_mask, src, src_mask, training):
        # masked self attention masked 自注意力 q k v都是预测值 还需要对预测值进行masked
        _trg, _ = self.self_attention.forward(trg, trg, trg, trg_mask, training)
        trg = self.self_attention_norm.forward(trg + self.dropout.forward(_trg, training))
        # cross attention 交叉注意力 query是预测值，k,v是原目标
        _trg, attention = self.encoder_attention.forward(trg, src, src, src_mask, training)
        trg = self.enc_attn_layer_norm.forward(trg + self.dropout.forward(_trg, training))

        # feed forward
        _trg = self.position_wise_feed_forward.forward(trg, training)
        # add & Norm
        trg = self.ff_layer_norm.forward(trg + self.dropout.forward(_trg, training))

        return trg, attention

    def backward(self, error):
        # feed forward的norm&add 反向传播
        error = self.ff_layer_norm.backward(error)
        # feed forward的反向传播
        _error = self.position_wise_feed_forward.backward(self.dropout.backward(error))
        # Cross Attention的Norm&Add 既输出给了position_wise_feed_forward 也通过残差连接输出给了ff_layer_norm
        # 所以这里的反向传播 error = error + _error
        error = self.enc_attn_layer_norm.backward(error + _error)
        # Cross Attention的自注意力delta要分开,一部分进入encoder,一部分继续在decoder中反向传播
        # Q K V的delta 对应_error在decoder继续传播  enc_error1, enc_error2 需要传递到编码器
        _error, enc_error1, enc_error2 = self.encoder_attention.backward(self.dropout.backward(error))

        # masked 注意力的norm&Add层 取的是enc_attn_layer_norm的delta 和 Q矩阵的delta
        error = self.self_attention_norm.backward(error + _error)
        # masked attention 的 q k v 误差,需要全部反向传播给下一个的decoder_layer
        _error, _error2, _error3 = self.self_attention.backward(self.dropout.backward(error))
        """
        结合架构图,我们给出结论,在transformer中的反向传播,所有forward多箭头的特征传递,反向传播均需加和计算
        """
        return _error +_error2 + _error3 + error, enc_error1 + enc_error2

    def set_optimizer(self, optimizer):
        self.self_attention_norm.set_optimizer(optimizer)
        self.enc_attn_layer_norm.set_optimizer(optimizer)
        self.ff_layer_norm.set_optimizer(optimizer)
        self.self_attention.set_optimizer(optimizer)
        self.encoder_attention.set_optimizer(optimizer)
        self.position_wise_feed_forward.set_optimizer(optimizer)

    def update_weights(self, layer_num):
        layer_num = self.self_attention_norm.update_weights(layer_num)
        layer_num = self.enc_attn_layer_norm.update_weights(layer_num)
        layer_num = self.ff_layer_norm.update_weights(layer_num)
        layer_num = self.self_attention.update_weights(layer_num)
        layer_num = self.encoder_attention.update_weights(layer_num)
        layer_num = self.position_wise_feed_forward.update_weights(layer_num)

        return layer_num

