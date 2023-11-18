import sys, os
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)

import numpy as np


class DataPreparator():

    def __init__(self, tokens, indexes):

        self.PAD_TOKEN = tokens[0]
        self.SOS_TOKEN = tokens[1]
        self.EOS_TOKEN = tokens[2]
        self.UNK_TOKEN = tokens[3]

        self.PAD_INDEX = indexes[0]
        self.SOS_INDEX = indexes[1]
        self.EOS_INDEX = indexes[2]
        self.UNK_INDEX = indexes[3]

        self.toks_and_inds = {self.PAD_TOKEN: self.PAD_INDEX, self.SOS_TOKEN: self.SOS_INDEX, self.EOS_TOKEN: self.EOS_INDEX, self.UNK_TOKEN: self.UNK_INDEX}
        # 语料表
        self.vocabs = None

    def prepare_data(self, path = 'dataset/', batch_size = 1, min_freq = 10):
        """
        path（默认值为'dataset/'）：指定数据集的路径。该参数用于指定数据集所在的目录。
        batch_size（默认值为1）：批处理大小。它指定了在训练过程中每个批次中所包含的样本数量。较大的批量大小可能会加快训练速度，但也可能导致内存消耗增加。
        min_freq（默认值为10）：最小词频。该参数用于过滤掉低频词。词频是指在整个数据集中出现的次数。通过设置min_freq，可以排除那些在数据集中出现次数过少的词语，以减少噪声和提高模型性能。
        """
        # 1 =====================================================================
        # 将训练数据 测试数据 验证数据三个数据集的英文和德文加载到字典里
        train_data, val_data, test_data = self.import_multi30k_dataset(path)
        # 2 =====================================================================
        # 数据处理 清除特殊字符 并按照空格进行分隔
        train_data, val_data, test_data = self.clear_dataset(train_data, val_data, test_data)
        print(f"train data sequences num = {len(train_data)}")
        # 3 =====================================================================
        # 根据分词构建<语料表>
        self.vocabs = self.build_vocab(train_data, self.toks_and_inds, min_freq)
        print(f"EN vocab length = {len(self.vocabs[0])}; DE vocab length = {len(self.vocabs[1])}")
        # 4 =====================================================================
        # 添加token标记，这个标记类似bert的cls和sep 批次大小32，对批次中的数据进行<pad>补充，并按batch切分句子
        train_data = self.add_tokens(train_data, batch_size)
        print(f"batch num = {len(train_data)}")
        # 根据语料表构建训练集词向量
        train_source, train_target = self.build_dataset(train_data, self.vocabs)
        # 5 =====================================================================
        test_data = self.add_tokens(test_data, batch_size)
        test_source, test_target = self.build_dataset(test_data, self.vocabs)

        val_data = self.add_tokens(val_data, batch_size)
        val_source, val_target = self.build_dataset(val_data, self.vocabs)
        # 4 =====================================================================
        return (train_source, train_target), (test_source, test_target), (val_source, val_target)

    def get_vocabs(self):
        return self.vocabs

    def filter_seq(self, seq):
        chars2remove = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'

        return ''.join([c for c in seq if c not in chars2remove])

    def lowercase_seq(self, seq):
        return seq.lower()


    def import_multi30k_dataset(self, path):

        ret = []
        filenames = ["train", "val", "test"]

        for filename in filenames:

            examples = []

            en_path = os.path.join(path, filename + '.en')
            de_path = os.path.join(path, filename + '.de')

            en_file = [l.strip() for l in open(en_path, 'r', encoding='utf-8')]
            de_file = [l.strip() for l in open(de_path, 'r', encoding='utf-8')]

            assert len(en_file) == len(de_file)

            for i in range(len(en_file)):
                if en_file[i] != '' and de_file[i] != '':
                    en_seq, de_seq = en_file[i], de_file[i]
                    """
                    {'en': en seq1, 'de': de seq1}
                    {'en': en seq2, 'de': de seq2}
                    {'en': en seq3, 'de': de seq3}
                    """
                    examples.append({'en': en_seq, 'de': de_seq})
            """
            [
                [{'en': en_seq1, 'de': de_seq1}
                {'en': en_seq2, 'de': de_seq2}
                {'en': en_seq3, 'de': de_seq3}]
                ,
                 [{'en': en_seq1, 'de': de_seq1}
                {'en': en_seq2, 'de': de_seq2}
                {'en': en_seq3, 'de': de_seq3}]
                , 
                [{'en': en_seq1, 'de': de_seq1}
                {'en': en_seq2, 'de': de_seq2}
                {'en': en_seq3, 'de': de_seq3}]
            ]
            """
            ret.append(examples)

        return tuple(ret)


    def clear_dataset(self, *data):

        for dataset in data:
            for example in dataset:
                # 过滤特殊字符
                example['en'] = self.filter_seq(example['en'])
                example['de'] = self.filter_seq(example['de'])
                # 不区分大小写
                example['en'] = self.lowercase_seq(example['en'])
                example['de'] = self.lowercase_seq(example['de'])
                # 将英文和德文句子根据空格进行分词，将分词结果作为列表更新
                example['en'] = example['en'].split()
                example['de'] = example['de'].split()

        return data



    def build_vocab(self, dataset, toks_and_inds, min_freq = 1):
        """

        :param dataset: 包含示例数据的数据集
        :param toks_and_inds: 一个包含已知标记和索引的词汇表字典
        :param min_freq: 指定单词在数据集中出现的最小频率，低于该频率的单词将被过滤掉
        :return: [en_vocab]英文词汇表，包含单词和对应的索引。[de_vocab]：德文词汇表，包含单词和对应的索引。
        """

        en_vocab = toks_and_inds.copy(); en_vocab_freqs = {}
        de_vocab = toks_and_inds.copy(); de_vocab_freqs = {}
        # 语料表仅包含训练集的数据
        for example in dataset:
            for word in example['en']:
                if word not in en_vocab_freqs:
                    en_vocab_freqs[word] = 0
                en_vocab_freqs[word] += 1
            for word in example['de']:
                if word not in de_vocab_freqs:
                    de_vocab_freqs[word] = 0
                de_vocab_freqs[word] += 1

        for example in dataset:
            for word in example['en']:
                if word not in en_vocab and en_vocab_freqs[word] >= min_freq:
                    en_vocab[word] = len(en_vocab)
            for word in example['de']:
                if word not in de_vocab and de_vocab_freqs[word] >= min_freq:
                    de_vocab[word] = len(de_vocab)
        # 语料表中的词的词频最少为2，小于2的不加入词频
        return en_vocab, de_vocab


    def add_tokens(self, dataset, batch_size):
        for example in dataset:
            # 在每个示例的英文和德文序列前后添加开始和结束标记 表明一个Segment的起始与结束
            example['en'] = [self.SOS_TOKEN] + example['en'] + [self.EOS_TOKEN]
            example['de'] = [self.SOS_TOKEN] + example['de'] + [self.EOS_TOKEN]
        # 将数据集划分为多个批次 每个批次大小为batch_size
        data_batches = np.array_split(dataset, np.arange(batch_size, len(dataset), batch_size))

        for batch in data_batches:
            # 每个批次中最长的德文和英文句子的具体长度
            max_en_seq_len, max_de_seq_len = 0, 0
            # 找出每个批次最长的句子
            for example in batch:
                max_en_seq_len = max(max_en_seq_len, len(example['en']))
                max_de_seq_len = max(max_de_seq_len, len(example['de']))
            # 训练需要归一化数据维度，每个批次中不足最大长度的句子统一<pad>补充
            for example in batch:
                example['en'] = example['en'] + [self.PAD_TOKEN] * (max_en_seq_len - len(example['en']))
                example['de'] = example['de'] + [self.PAD_TOKEN] * (max_de_seq_len - len(example['de']))

        """
        比如一个batch为2的句子
         i love  you   ====》我 爱 你
        hi             ====》你 好
        经过padding后变成
         i love  you   ====》我 爱 你
        hi <pad> <pad> ====》你 好 <pad>
        
        """
        return data_batches


    def build_dataset(self, dataset, vocabs):
        
        source, target = [], []
        # 32个句子
        for batch in dataset:
            # 一个句子 经过添加token及分词后的处理
            source_tokens, target_tokens = [], []
            for example in batch:
                en_inds = [vocabs[0][word] if word in vocabs[0] else self.UNK_INDEX for word in example['en']]
                de_inds = [vocabs[1][word] if word in vocabs[1] else self.UNK_INDEX for word in example['de']]
                # 将词片转成词向量 对于不在语料表中的词转成self.UNK_INDEX
                source_tokens.append(en_inds)
                target_tokens.append(de_inds)

            source.append(np.asarray(source_tokens))
            target.append(np.asarray(target_tokens))

        return source, target
