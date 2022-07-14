class Image_Text_Attention(Layer):
	# 该层的输入有三部分image_emb、seq_emb、mask
	# image_emb: [None, M, 4096]对应M个4096维的图像向量(由vgg16提取得到)，每条评论的M可以不一致
    # seq_emb:   [None, L, k]表示L个维度为k的句向量
    # mask:      [None, L]表示L个句子的mask(因为存在句子数不足L的文档，有被padding的句子)
    # output:    [None, M, k]输出为M个图像对应的文档向量表示
    def __init__(self, dropout_rate=0.0):
        super(Image_Text_Attention, self).__init__()
        self.dropout_layer = Dropout(dropout_rate)

    def build(self, input_shape):
        self.l = input_shape[1][1]   # 句子个数
        self.k = input_shape[1][-1]  # 句向量维度
        self.img_layer = Dense(1, activation='tanh', use_bias=True)  # 将image_emb映射到1维
        self.seq_layer = Dense(1, activation='tanh', use_bias=True)  # 将seq_emb也映射到1维(方便内积)
        self.V_weight = self.add_weight(name='V', shape=(self.l, self.l),
                                        initializer=tf.keras.initializers.glorot_uniform(),
                                        trainable=True)

    def call(self, inputs, **kwargs):
        image_emb, seq_emb, mask = inputs  # 输入为三部分[image_emb, seq_emb, mask]

        # 线性映射
        p = self.img_layer(image_emb)  # [None, M, 1]
        q = self.seq_layer(seq_emb)    # [None, L, 1]

        # 内积+映射(计算score)
        emb = tf.matmul(p, q, transpose_b=True)   # [None, M, L]
        emb = emb + tf.transpose(q, [0, 2, 1])    # [None, M, L]
        emb = tf.matmul(emb, self.V_weight)       # [None, M, L]
        score = self.dropout_layer(emb)           # 随机dropout(也可不要)

        # mask
        mask = tf.tile(tf.expand_dims(mask, axis=1), [1, score.shape[1], 1])  # [None, M, L]，将mask矩阵复制到与score相同的形状
        padding = tf.cast(tf.ones_like(mask) * (-2 ** 31 + 1), tf.float32)
        score = tf.where(tf.equal(mask, 0), padding, score)
        score = tf.nn.softmax(score, axis=-1)      # [None, M, L]

        # 向量加权和
        output = tf.matmul(score, seq_emb)   # [None, M, k]
        output /= self.k**0.5                # 归一化
        return output
