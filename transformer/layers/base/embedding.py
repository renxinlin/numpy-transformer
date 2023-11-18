try:
    import cupy as np
    is_cupy_available = True
except:
    import numpy as np
    is_cupy_available = False


class Embedding():
    """
    Add Embedding layer
    ---------------
        Args:
            `input_dim`: (int), size of vocabulary
            `output_dim` (int): number of neurons in the layer (vector size)
        Returns:
            input: data with shape (batch_size, input_length)
            output: data with shape (batch_size, input_length, output_dim)
    """

    def __init__(self, input_dim, output_dim, data_type = np.float32):
        self.input_dim = input_dim
        self.output_dim   = output_dim
        # w大小 语料表size*256
        self.w = None

        self.optimizer = None
        self.data_type = data_type
        # 从正态分布中生成符合一定均值和标准差要求的随机数，用于初始化神经网络的权重矩阵
        self.build()

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def build(self):
        """
        基于正太分布随机初始化w
        :return:
        """
        self.w = np.random.normal(0, pow(self.input_dim, -0.5), (self.input_dim, self.output_dim)).astype(self.data_type)

        self.v, self.m         = np.zeros_like(self.w).astype(self.data_type), np.zeros_like(self.w).astype(self.data_type)
        self.v_hat, self.m_hat = np.zeros_like(self.w).astype(self.data_type), np.zeros_like(self.w).astype(self.data_type)


    #one hot encoding
    def prepare_labels(self, batch_labels):
        # batch*sequence大小
        batch_labels = batch_labels.astype(np.int32)
        # batch * sequence * 语料库大小
        prepared_batch_labels = np.zeros((batch_labels.size,  self.input_dim)) #batch_labels.max() + 1)
        # batch * sequence * 的词进行one-hot编码
        """
            假设
            batch_labels = [[10, 2], [4, 19]]
            input_dim =20
            这句one-hot编码实现
            [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
             [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
             [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
             [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]
        """
        prepared_batch_labels[np.arange(batch_labels.size), batch_labels.reshape(1, -1)] = 1
        # batch * sequence * 语料库大小 张量
        return prepared_batch_labels.reshape(self.batch_size, self.current_input_length, self.input_dim).astype(self.data_type)


    def forward(self, X):
        self.input_data = X # (batch_size, input_length); inputs: values of vocabulary from 0 to input_dim - 1
        
        if not all([np.equal(len(self.input_data[0]), len(arr)).all() for arr in self.input_data]):
            raise ValueError("Input sequences must be of the same length")
        # 句子大小
        self.current_input_length = len(self.input_data[0])
        self.batch_size = len(self.input_data)
        # 从语料库二维转词向量embedding 为3维采用独热编码
        self.input_data = self.prepare_labels(self.input_data)
        #     input_data（batch * segment size * 语料表sizw）     w（语料表size*256）
        self.output_data = np.dot(self.input_data, self.w)
        # 输出 batch * segment size * 256
        return self.output_data

    def backward(self, error):
        """
        前向 X*W 反向 delta * X
        :param error:
        :return:
        """
        self.grad_w = np.matmul(np.transpose(self.input_data, axes = (0, 2, 1)), error).sum(axis = 0)

        # output_error = np.dot(error, self.w.T)

        # return output_error
        return None

    def update_weights(self, layer_num):
        self.w, self.v, self.m, self.v_hat, self.m_hat  = self.optimizer.update(self.grad_w, self.w, self.v, self.m, self.v_hat, self.m_hat, layer_num)

        return layer_num + 1
    def get_grads(self):
        return self.grad_w, self.grad_b

    def set_grads(self, grads):
        self.grad_w, self.grad_b = grads
        
