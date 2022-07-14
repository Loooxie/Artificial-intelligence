import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
import tensorflow.keras.backend as K

# 自注意力层
class Self_Attention(Layer):
    # input:  [None, n, k]输入为n个维度为k的词向量
    # mask:   [None, n]表示填充词位置的mask
    # output: [None, k]输出n个词向量的加权和
    def __init__(self, dropout_rate=0.0):
        super(Self_Attention, self).__init__()
        self.dropout_layer = Dropout(dropout_rate)

    def build(self, input_shape):
        self.k = input_shape[0][-1]  #词向量维度
        self.W_layer = Dense(self.k, activation='tanh', use_bias=True) #对h的映射
        self.U_weight = self.add_weight(name='U', shape=(self.k, 1),   #U记忆矩阵
                                        initializer=tf.keras.initializers.glorot_uniform(),
                                        trainable=True)

    def call(self, inputs, **kwargs):
        input, mask = inputs #输入有两部分[input, mask]
        if K.ndim(input) != 3:
            raise ValueError("The dim of inputs is required 3 but get {}".format(K.ndim(input)))

        # 计算score
        x = self.W_layer(input)              # [None, n, k]
        score = tf.matmul(x, self.U_weight)  # [None, n, 1]
        score = self.dropout_layer(score)    # 随机dropout(也可不要)

        # softmax之前进行mask
        mask = tf.expand_dims(mask, axis=-1)  # [None, n, 1]
        padding = tf.cast(tf.ones_like(mask)*(-2**31+1), tf.float32) #mask的位置填充很小的负数
        score = tf.where(tf.equal(mask, 0), padding, score)
        score = tf.nn.softmax(score, axis=1)  # [None, n, 1] mask之后计算softmax

        # 向量加权和
        output = tf.matmul(input, score, transpose_a=True)   # [None, k, 1]
        output /= self.k**0.5                                # 归一化
        output = tf.squeeze(output, axis=-1)                 # [None, k]
        return output
