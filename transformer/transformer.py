import sys, os
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)


import numpy as np
try:
    import cupy as cp
    is_cupy_available = True
    print('CuPy is available. Using CuPy for all computations.')
except:
    is_cupy_available = False
    print('CuPy is not available. Switching to NumPy.')

import pickle as pkl
from tqdm import tqdm
from transformer.modules import Encoder
from transformer.modules import Decoder
from transformer.optimizers import Adam, Nadam, Momentum, RMSProp, SGD, Noam
from transformer.losses import CrossEntropy
from transformer.prepare_data import DataPreparator
import matplotlib.pyplot as plt



DATA_TYPE = np.float32
BATCH_SIZE = 32

PAD_TOKEN = '<pad>' # pad标记,补齐填充
SOS_TOKEN = '<sos>' # 开始标记
EOS_TOKEN = '<eos>' # 结束标记
UNK_TOKEN = '<unk>' # 未能识别的单词或标记
# 索引位
PAD_INDEX = 0
SOS_INDEX = 1
EOS_INDEX = 2
UNK_INDEX = 3

# 数组化
tokens  = (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN)
indexes = (PAD_INDEX, SOS_INDEX, EOS_INDEX, UNK_INDEX)
# 数据准备器，初始化tokens标记
data_preparator = DataPreparator(tokens, indexes)
# 根据数据集构建语料表 ，根据语料表索引构建训练集，测试集，验证集词向量
train_data, test_data, val_data = data_preparator.prepare_data(
    path = '../dataset/',
    batch_size = BATCH_SIZE,
    min_freq = 2)
""""""""""""""""""""""""""""""
# 训练中，每个批次的句子对应的词向量大小由最大句子长度决定
""""""""""""""""""""""""""""""

source, target = train_data
# 获取语料表
""" en/de
<pad> 0
<sos> 1
<eos> 2
<unk> 3
i     4
hello 5
:
:
:
end   n-1
"""
train_data_vocabs = data_preparator.get_vocabs()


# transformer模型定义
class Seq2Seq():

    def __init__(self, encoder, decoder, pad_idx) -> None:
        # 基于自注意力的编码器
        self.encoder = encoder
        # 基于mask和cross注意力的编码器
        self.decoder = decoder
        self.pad_idx = pad_idx
        # Adam自适应梯度下降优化器
        self.optimizer = Adam()
        # 采用交叉熵损失
        self.loss_function = CrossEntropy()

    def set_optimizer(self):
        # 两者优化器相同
        encoder.set_optimizer(self.optimizer)
        decoder.set_optimizer(self.optimizer)

    def compile(self, optimizer, loss_function):
        # 设置优化器和损失函数
        self.optimizer = optimizer
        self.loss_function = loss_function


    def load(self, path):
        """
        模型反序列化加载:[一般微调时加载]
        :param path:
        :return:
        """
        pickle_encoder = open(f'{path}/encoder.pkl', 'rb')
        pickle_decoder = open(f'{path}/decoder.pkl', 'rb')

        self.encoder = pkl.load(pickle_encoder)
        self.decoder = pkl.load(pickle_decoder)

        pickle_encoder.close()
        pickle_decoder.close()

        print(f'Loaded from "{path}"')

    def save(self, path):
        """
        模型序列化存储: 一般预训练进行存储
        :param path: 存储路径
        :return: None
        """
        if not os.path.exists(path):
            os.makedirs(path)

        pickle_encoder = open(f'{path}/encoder.pkl', 'wb')
        pickle_decoder = open(f'{path}/decoder.pkl', 'wb')

        pkl.dump(self.encoder, pickle_encoder)
        pkl.dump(self.decoder, pickle_decoder)

        pickle_encoder.close()
        pickle_decoder.close()

        print(f'Saved to "{path}"')

    def get_pad_mask(self, x):
        # x长度为 (batch_size, seq_len) 一个批次的所有句子
        # 生成填充遮罩，将不等于self.pad_idx的位置设为1，等于self.pad_idx的位置设为0
        # 比如 [1,2,0]=>[1,1,0]
        return (x != self.pad_idx).astype(int)[:, np.newaxis, :]

    def get_sub_mask(self, x):
        #x: (batch_size, seq_len)
        seq_len = x.shape[1]
        subsequent_mask = np.triu(np.ones((seq_len, seq_len)), k = 1).astype(int)
        subsequent_mask = np.logical_not(subsequent_mask)
        """
         seq_len = 5 返回如下
         其用法理解下self attention 的masked 掩码机制
         每个词只能看到自己和历史上文,不具备理解下文和未来的能力
         [[ True False False False False]
         [ True  True False False False]
         [ True  True  True False False]
         [ True  True  True  True False]
         [ True  True  True  True  True]]
         """
        return subsequent_mask

    def forward(self, src, trg, training):
        """
        前向传播进行训练
        :param src:
        :param trg:
        :param training:
        :return:
        """
        src, trg = src.astype(DATA_TYPE), trg.astype(DATA_TYPE)
        #src: (batch_size, source_seq_len)
        #tgt: (batch_size, target_seq_len)

        # src_mask: (batch_size, 1, seq_len)
        # tgt_mask: (batch_size, seq_len, seq_len)
        """
            这两次的掩码,对于input来说,padding填充不应该参与注意力预测
            output的masked掩码为出自论文指导,
            
        """
        # padding 掩码
        src_mask = self.get_pad_mask(src)
        # padding 掩码 & masked掩码
        trg_mask = self.get_pad_mask(trg) & self.get_sub_mask(trg)
        # encode进行自注意力+残差+正则+FC (8头3层)
        enc_src = self.encoder.forward(src, src_mask, training)
        # 编码器的输出作为解码器的K&V矩阵进行计算
        out, attention = self.decoder.forward(trg, trg_mask, enc_src, src_mask, training)
        # output: (batch_size, target_seq_len, vocab_size)
        # attn: (batch_size, heads_num, target_seq_len, source_seq_len)
        # 解码器的输出
        return out, attention

    def backward(self, error):
        """
        反向传播求损失和梯度
        :param error:
        :return:
        """
        error = self.decoder.backward(error)
        # 所有的decoder的 Cross Attention的K,V矩阵的delta作为encoder的起始delta进行反向传播
        error = self.encoder.backward(self.decoder.encoder_error)

    def update_weights(self):
        """
        更新权重
        :return:
        """
        self.encoder.update_weights()
        self.decoder.update_weights()

    def _train(self, source, target, epoch, epochs):
        loss_history = []

        tqdm_range = tqdm(enumerate(zip(source, target)), total = len(source))
        for batch_num, (source_batch, target_batch) in tqdm_range:
            # outout大小为batch*sequence size * 语料表大小
            output, attention = self.forward(source_batch, target_batch[:,:-1], training = True)

            _output = output.reshape(output.shape[0] * output.shape[1], output.shape[2])
            # 计算交叉熵损失
            loss_history.append(self.loss_function.loss(_output, target_batch[:, 1:].astype(np.int32).flatten()).mean())#[:, np.newaxis]
            # 计算梯度
            error = self.loss_function.derivative(_output, target_batch[:, 1:].astype(np.int32).flatten())#[:, np.newaxis]

            # 反向计算梯队与反向传播
            self.backward(error.reshape(output.shape))
            # 更新权重 [backward 每个class都有,update时针对有W和B的class进行update]
            self.update_weights()
            #
            tqdm_range.set_description(
                f"training | loss: {loss_history[-1]:.7f} | perplexity: {np.exp(loss_history[-1]):.7f} | epoch {epoch + 1}/{epochs}" #loss: {loss:.4f}
            )

            if batch_num == (len(source) - 1):
                if is_cupy_available:
                    epoch_loss = cp.mean(cp.array(loss_history))
                else:
                    epoch_loss = np.mean(loss_history)

                tqdm_range.set_description(
                    f"training | avg loss: {epoch_loss:.7f} | avg perplexity: {np.exp(epoch_loss):.7f} | epoch {epoch + 1}/{epochs}"
                )

        return epoch_loss.get() if is_cupy_available else epoch_loss

    def _evaluate(self, source, target):
        # 用于存储每个批次的损失值
        loss_history = []
        # 使用tqdm创建一个进度条，显示评估的进度
        tqdm_range = tqdm(enumerate(zip(source, target)), total = len(source))
        # 预测，并计算损失
        for batch_num, (source_batch, target_batch) in tqdm_range:
            # 输出预测结果
            output, attention = self.forward(source_batch, target_batch[:,:-1], training = False)
            # 批次*序列长度*词向量大小（词汇表大小） one-hot独热编码 变成一维向量，便于计算损失
            _output = output.reshape(output.shape[0] * output.shape[1], output.shape[2])
            # 计算并记录损失值
            loss_history.append(self.loss_function.loss(_output, target_batch[:, 1:].astype(np.int32).flatten()).mean())
            # 在进度条中显示当前批次的损失和困惑度（困惑度越低，模型的性能就越好）
            tqdm_range.set_description(
                f"testing  | loss: {loss_history[-1]:.7f} | perplexity: {np.exp(loss_history[-1]):.7f}"
            )
            # 最后一个批次预测后计算损失均值
            if batch_num == (len(source) - 1):
                if is_cupy_available:
                    epoch_loss = cp.mean(cp.array(loss_history))
                else:
                    epoch_loss = np.mean(loss_history)
                # 在进度条中显示整个评估集的平均损失和困惑度
                tqdm_range.set_description(
                    f"testing  | avg loss: {epoch_loss:.7f} | avg perplexity: {np.exp(epoch_loss):.7f}"
                )
        # 返回整个评估集的平均损失值
        return epoch_loss.get() if is_cupy_available else epoch_loss


    def fit(self, train_data, val_data, epochs, save_every_epochs, save_path = None, validation_check = False):
        self.set_optimizer()

        best_val_loss = float('inf')

        train_loss_history = []
        val_loss_history = []

        train_source, train_target = train_data
        val_source, val_target = val_data
        # epochs次训练
        for epoch in range(epochs):
            # 训练
            train_loss_history.append(self._train(train_source, train_target, epoch, epochs))
            # 验证
            val_loss_history.append(self._evaluate(val_source, val_target))


            if (save_path is not None) and ((epoch + 1) % save_every_epochs == 0):
                if validation_check == False:
                    self.save(save_path + f'/{epoch + 1}')
                else:
                    if val_loss_history[-1] < best_val_loss:
                        best_val_loss = val_loss_history[-1]

                        self.save(save_path + f'/{epoch + 1}')
                    else:
                        print(f'Current validation loss is higher than previous; Not saved')
        # 训练完成后返回损失
        return train_loss_history, val_loss_history




    def predict(self, sentence, vocabs, max_length = 50):
        """
        :param sentence:  输入语句
        :param vocabs:  语料表
        :param max_length: 生成式输出最大长度
        :return: 预测结果以及注意力分布
        """
        # 输入转词向量 单个句子这里不进行padding
        src_inds = [vocabs[0][word] if word in vocabs[0] else UNK_INDEX for word in sentence]
        src_inds = [SOS_INDEX] + src_inds + [EOS_INDEX]

        # 获取mask结果
        src = np.asarray(src_inds).reshape(1, -1)
        src_mask =  self.get_pad_mask(src)

        enc_src = self.encoder.forward(src, src_mask, training = False)
        # 第一个输出是已知的标记，其他的输出通过生成获取
        trg_inds = [SOS_INDEX]

        for _ in range(max_length):
            trg = np.asarray(trg_inds).reshape(1, -1)
            trg_mask = self.get_pad_mask(trg) & self.get_sub_mask(trg)

            out, attention = self.decoder.forward(trg, trg_mask, enc_src, src_mask, training = False)
            # 对decoder的输出进行预测
            trg_indx = out.argmax(axis=-1)[:, -1].item()
            trg_inds.append(trg_indx)
            # 判断是否结束,没结束,将输入与新的输出重新推送给decoder进行一个词的预测
            if trg_indx == EOS_INDEX or len(trg_inds) >= max_length:
                break
        # 通过语料表将预测词转人类可读
        reversed_vocab = dict((v,k) for k,v in vocabs[1].items())
        decoded_sentence = [reversed_vocab[indx] if indx in reversed_vocab else UNK_TOKEN for indx in trg_inds]
        return decoded_sentence[1:], attention[0]



# 编码语料表长度
INPUT_DIM = len(train_data_vocabs[0])
# 解码语料表长度
OUTPUT_DIM = len(train_data_vocabs[1])
# # 定义隐藏层维度
HID_DIM = 256  #512 in original paper
# 编解码器层数
ENC_LAYERS = 3 #6 in original paper
DEC_LAYERS = 3 #6 in original paper
# 多头数量
ENC_HEADS = 8
DEC_HEADS = 8
# 全连接层大小
FF_SIZE = 512  #2048 in original paper
# DROPOUT比例
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
# 定义最大句子长度
MAX_LEN = 5000


encoder = Encoder(INPUT_DIM, ENC_HEADS, ENC_LAYERS, HID_DIM, FF_SIZE, ENC_DROPOUT, MAX_LEN, DATA_TYPE)
decoder = Decoder(OUTPUT_DIM, DEC_HEADS, DEC_LAYERS, HID_DIM, FF_SIZE, DEC_DROPOUT, MAX_LEN, DATA_TYPE)



model = Seq2Seq(encoder, decoder, PAD_INDEX)

try:
    model.load("saved models/seq2seq_model/0")
except:
    print("Can't load saved model state")


# 模型进行编译，包括设置优化器（Noam优化器）和损失函数（交叉熵损失）。
model.compile(
    optimizer = Noam(
        Adam(alpha = 1e-4, beta = 0.9, beta2 = 0.98, epsilon = 1e-9), #NOTE: alpha doesn`t matter for Noam scheduler
        model_dim = HID_DIM,
        scale_factor = 2,
        warmup_steps = 4000
    )
    , loss_function = CrossEntropy(ignore_index=PAD_INDEX)
)
train_loss_history, val_loss_history = None, None
# 训练模型
train_loss_history, val_loss_history = model.fit(train_data, val_data, epochs = 30, save_every_epochs = 5, save_path = "saved models/seq2seq_model", validation_check = True)# "saved models/seq2seq_model"


# 完成训练后进行绘图
def plot_loss_history(train_loss_history, val_loss_history):
    plt.plot(train_loss_history)
    plt.plot(val_loss_history)
    plt.title('Loss history')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

if train_loss_history is not None and val_loss_history is not None:
    plot_loss_history(train_loss_history, val_loss_history)


# 选择一部分验证集数据验证
_, _, val_data = data_preparator.import_multi30k_dataset(path = "dataset/")
val_data = data_preparator.clear_dataset(val_data)[0]
sentences_num = 10

random_indices = np.random.randint(0, len(val_data), sentences_num)
sentences_selection = [val_data[i] for i in random_indices]

#Translate sentences from validation set
for i, example in enumerate(sentences_selection):
    print(f"\nExample №{i + 1}")
    print(f"Input sentence: { ' '.join(example['en'])}")
    print(f"Decoded sentence: {' '.join(model.predict(example['en'], train_data_vocabs)[0])}")
    print(f"Target sentence: {' '.join(example['de'])}")




def plot_attention(sentence, translation, attention, heads_num = 8, rows_num = 2, cols_num = 4):
    # 确保指定的行数和列数与头数相匹配
    assert rows_num * cols_num == heads_num
    # 在句子开头和结尾添加特殊标记（如起始标记和终止标记），这有助于可视化
    sentence = [SOS_TOKEN] + [word.lower() for word in sentence] + [EOS_TOKEN]
    # 创建一个新的图表
    fig = plt.figure(figsize = (15, 25))
    # 每个自注意力都绘制出来
    for h in range(heads_num):
        # 在图表中添加子图
        ax = fig.add_subplot(rows_num, cols_num, h + 1)
        ax.set_xlabel(f'Head {h + 1}')
        # 绘制注意力权重矩阵
        if is_cupy_available:
            ax.matshow(cp.asnumpy(attention[h]), cmap = 'inferno')
        else:
            ax.matshow(attention[h], cmap = 'inferno')
        # 设置坐标轴标签字体大小
        ax.tick_params(labelsize = 7)
        # 设置x和y轴标签
        ax.set_xticks(range(len(sentence)))
        ax.set_yticks(range(len(translation)))
        # 设置x轴标签为源语言句子单词，并进行90度旋转
        ax.set_xticklabels(sentence, rotation=90)
        # 设置y轴标签为目标语言句子单词
        ax.set_yticklabels(translation)


    plt.show()

# 用一个自定义的句子进行翻译测试效果
#Plot Attention
sentence = sentences_selection[0]['en']#['a', 'trendy', 'girl', 'talking', 'on', 'her', 'cellphone', 'while', 'gliding', 'slowly', 'down', 'the', 'street']
print(f"\nInput sentence: {sentence}")
decoded_sentence, attention =  model.predict(sentence, train_data_vocabs)
print(f"Decoded sentence: {decoded_sentence}")

plot_attention(sentence, decoded_sentence, attention)