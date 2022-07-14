class VggNet(Layer):
    def __init__(self, block_nums, out_dim=1000, dropout_rate=0.0):
    	# block_nums: [list]，表示每个模块中连续卷积的个数，vgg16为[2,2,3,3,3]
    	# out_dim: 该层最终的输出维度
        super(VggNet, self).__init__()
        self.cnn_block1 = self.get_cnn_block(64, block_nums[0])
        self.cnn_block2 = self.get_cnn_block(128, block_nums[1])
        self.cnn_block3 = self.get_cnn_block(256, block_nums[2])
        self.cnn_block4 = self.get_cnn_block(512, block_nums[3])
        self.cnn_block5 = self.get_cnn_block(512, block_nums[4])
        self.out_block = self.get_out_block([4096, 4096], out_dim, dropout_rate)
        self.flatten = Flatten()

    # 单个卷积模块的搭建(layer_num个连续卷积加一个池化)
    def get_cnn_block(self, out_channel, layer_num):
        layer = []
        for i in range(layer_num):
            layer.append(Conv2D(filters=out_channel,
                                kernel_size=3,
                                padding='same',
                                activation='relu'))
        layer.append(MaxPool2D(pool_size=(2,2), strides=2))
        return tf.keras.models.Sequential(layer) #封装成一个模块
        
	# 输出模块的搭建(连续的全连接层)
    def get_out_block(self, hidden_units, outdim, dropout_rate):
        layer = []
        for i in range(len(hidden_units)-1):
            layer.append(Dense(hidden_units[i], activation='relu'))
            layer.append(Dropout(dropout_rate))
        layer.append(Dense(outdim, activation='softmax'))
        return tf.keras.models.Sequential(layer) #封装成一个模块

    def call(self, inputs, **kwargs):
        # 标准输入：[batchsize, 224, 224, 3]
        if K.ndim(inputs) != 4:
            raise ValueError("The dim of inputs is required 4 but get {}".format(K.ndim(inputs)))

        x = inputs
        cnn_block_list = [self.cnn_block1, self.cnn_block2, self.cnn_block3, self.cnn_block4, self.cnn_block5]

        # 卷积层
        for cnn_block in cnn_block_list:
            x = cnn_block(x)
        x = self.flatten(x)

        # 输出层
        output = self.out_block(x)
        return output
