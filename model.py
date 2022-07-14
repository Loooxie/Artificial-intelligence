import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GRU, Bidirectional

class VistaNet(Model):
    def __init__(self, block_nums=[2,2,3,3,3], out_dim=4096, vgg_dropout=0.0, attention_dropout=0.0, gru_units=[64, 128], class_num=5):
    	# block_nums: vgg16各层卷积的个数
    	# out_dim: vgg16输出维度
    	# dropout: 各层的dropout系数
    	# gru_units: 两个单层双向GRU的输出维度
    	# class_num： 模型最终输出维度
        super(VistaNet, self).__init__()
        self.vgg16 = VggNet(block_nums, out_dim, vgg_dropout)       # VGG-16
        self.word_self_attention = Self_Attention(attention_dropout)# 第一层中的自注意力
        self.img_seq_attention = Image_Text_Attention(attention_dropout)  # 第二层中的Image-Text注意力
        self.doc_self_attention = Self_Attention(attention_dropout) # 第三层中的自注意力
        # 两个单层双向GRU层
        self.BiGRU_layer1 = Bidirectional(GRU(units=gru_units[0],
                                             kernel_regularizer=tf.keras.regularizers.l2(1e-5),
                                             recurrent_regularizer=tf.keras.regularizers.l2(1e-5),
                                             return_sequences=True),
                                          merge_mode='concat')
        self.BiGRU_layer2 = Bidirectional(GRU(units=gru_units[1],
                                             kernel_regularizer=tf.keras.regularizers.l2(1e-5),
                                             recurrent_regularizer=tf.keras.regularizers.l2(1e-5),
                                             return_sequences=True),
                                          merge_mode='concat')
        self.output_layer = Dense(class_num, activation='softmax') # 任务层

    def call(self, inputs, training=None, mask=None):
    	# 输入inputs包含三部分：(假设batchsize为1，省略掉第一维None)
        # image_inputs: [M, 227, 227, 3]一条评论样本包含的M个图像
        # text_inputs:  [L, T, k]一条样本表示一个文档，所以输入张量为3维:[最大句子数，最大单词数， 词向量维度]
        # mask: [L, T]每句话中mask词的位置
        image_inputs, text_inputs, mask = inputs 

        # 获取图像emb向量
        image_emb = self.vgg16(image_inputs)       # [M, 224, 224, 3] -> [M, 4096]

        # 经过GRU层获取词向量word_emb
        word_emb = self.BiGRU_layer1(text_inputs)  # [L, T, k] -> [L, T, 2k]

        # 经过self_attention得到句向量seq_emb
        input = [word_emb, mask]                   # [L, T, 2k] & [L, T]
        seq_emb = self.word_self_attention(input)  # [L, T, 2k] -> [L, 2k]

        # 经过GRU层提取语义
        input = tf.expand_dims(seq_emb, axis=0)    # [1, L, 2k]
        seq_emb = self.BiGRU_layer2(input)         # [1, L, 2k] -> [1, L, 4k]

        # 经过img_seq_attention得到M个文档向量doc_emb
        image_emb = tf.expand_dims(image_emb, axis=0) # [1, M, 4096]
        mask = tf.argmax(mask, axis=1)                # [L, ]
        mask = tf.expand_dims(mask, axis=0)           # [1, L]
        input = [image_emb, seq_emb, mask]
        doc_emb = self.img_seq_attention(input)       # [1, M, 4k] M个文档向量表示

        # 经过self_attention得到最终的文档向量
        mask = tf.ones(shape=[1, doc_emb.shape[1]])   # [1, M],全为非0值，因为该注意力无需mask
        input = [doc_emb, mask]
        D_emb = self.doc_self_attention(input)        # [1, 4k]

		# output layer
        output = self.output_layer(D_emb)             # [1, class_num]
        return output
