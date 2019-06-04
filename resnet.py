import tensorflow as tf


_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
DEFAULT_VERSION = 2
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES


def batch_norm(inputs, training, data_format):
    return tf.layers.batch_normalization(inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
                                         momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
                                         scale=True, training=training, fused=True)


def fixed_padding(inputs, kernel_size, data_format):
    """
    :param inputs: 텐서 [batch, height, width, chennls]
    :param kernel_size:  conv2d or max_pool2d 에 사용 될 값 양수
    :param data_format: 'channels_last' or 'channels_first'
    :return: input과 같은 형태의 tensor or padded (k > 1)
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(tensor=inputs,
                               paddings=[[0, 0], [0, 0], [pad_beg, pad_end],
                                         [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(tensor=inputs,
                               padding=[[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]])

    return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
    """Strided 2-D convolution with explicit padding"""
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)

    return tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format=data_format)


def _building_block_v2(inputs, filters, training, projection_shorcut, strides,
                       data_format):
    """
    :param inputs: tensor
    :param filters: 컨볼루션 필터 수
    :param training: training or inference mode --> batch norm 필요
    :param projection_shorcut: 숏컷 주로 1x1 컨볼루션 사용
    :param strides: 스트라이드 1보다크면 다운샘플됨
    :param data_format:
    :return: The output tensor of the block; shape should match inputs
    """
    if projection_shorcut is not None:
        shortcut = projection_shorcut(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=1,
        data_format=data_format)

    return inputs + shortcut


def _bottleneck_block(inputs, filters, training, projection_shorcut,
                      strides, data_format):
    shortcut = inputs
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    if projection_shorcut is not None:
        shortcut = projection_shorcut(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=1, strides=1,
        data_format=data_format)

    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=strides,
        data_format=data_format)

    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
        data_format=data_format)

    return inputs + shortcut


def block_layer(inputs, filters, bottleneck, block_fn, blocks, strides,
                training, name, data_format):
    """
    Create one layer of blocks for the ResNet model.
    :param inputs: tensor
    :param filters:
    :param bottleneck: Is the block created a bottleneck block.
    :param block_fn: The block to use within the model, either 'building_block' or 'bottleneck_block'
    :param blocks: THe number of blocks contained in the layer
    :param strides:
    :param training:
    :param name: A string name for the tensor output of the block layer.
    :param data_format:
    :return:
    """

    # Bottleneck blocks end with 4x the number of filters as they start with
    filters_out = filters * 4 if bottleneck else filters

    def projection_shortcut(inputs):
        return conv2d_fixed_padding(
            inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
            data_format=data_format)

    # Only the first block per block_layer uses projection_shortcut and strides
    inputs = block_fn(inputs, filters, training, projection_shortcut, strides, data_format)

    for _ in range(1, blocks):
        inputs = block_fn(inputs, filters, training, None, 1, data_format)

    return tf.identity(inputs, name)


class Model(object):

    def __init__(self, bottleneck, num_classes, num_filters,
                 kernel_size, conv_stride, first_pool_size, first_pool_stride,
                 block_sizes, block_strides,
                 resnet_version=DEFAULT_VERSION, data_format=None,
                 dtype=DEFAULT_DTYPE):
        """Creates a model for classifying an image.
        Args:
        :param bottleneck: Use regular blocks or bottleneck blocks.
        :param num_classes: The number of classes used as labels.
        :param num_filters: The number of filters to use for the first block layer
        of the model. This number is then doubled for each subsequent block layer.
        :param kernel_size: The kernel size to use for convolution.
        :param conv_stride: stride size for the initial convolutional layer
        :param first_pool_size: Pool size to be used for the first pooling layer.
            If none, the first pooling layer is skipped.
        :param first_pool_stride: stride size for the first pooling layer. Not used
            if first_pool_size is None.
        :param block_sizes: A list containing n values, where n is the number of sets of
            block layers desired. Each value should be the number of blocks in the
            i-th set.
        :param block_strides: List of integers representing the desired stride size for
            each of the sets of block layers. Should be same length as block_sizes.
        :param resnet_version: Integer representing which version of the ResNet network
            to use. See README for details. Valid values: [1, 2]
        :param data_format: Input format ('channels_last', 'channels_first', or None).
            If set to None, the format is dependent on whether a GPU is available.
        :param dtype: The TensorFlow dtype to use for calculations. If not specified
            tf.float32 is used.
        Raises:
          ValueError: if invalid version is selected.
        """

        if not data_format:
            data_format = ('channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

        self.bottleneck = bottleneck
        if bottleneck:
            self.block_fn = _bottleneck_block
        else:
            self.block_fn = _building_block_v2

        if dtype not in ALLOWED_TYPES:
            raise ValueError('dtype must be one of: {}'.format(ALLOWED_TYPES))

        self.data_format = data_format
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.conv_stride = conv_stride
        self.first_pool_size = first_pool_size
        self.first_pool_stride = first_pool_stride
        self.block_sizes = block_sizes
        self.block_strides = block_strides
        self.dtype = dtype
        self.pre_activation = resnet_version == 2

    def _custom_dtype_getter(self, getter, name, shape=None, dtype=DEFAULT_DTYPE,
                             *args, **kwargs):
        """Creates variables in fp32, then casts to fp16 if necessary.
        This function is a custom getter. A custom getter is a function with the
        same signature as tf.get_variable, except it has an additional getter
        parameter. Custom getters can be passed as the `custom_getter` parameter of
        tf.variable_scope. Then, tf.get_variable will call the custom getter,
        instead of directly getting a variable itself. This can be used to change
        the types of variables that are retrieved with tf.get_variable.
        The `getter` parameter is the underlying variable getter, that would have
        been called if no custom getter was used. Custom getters typically get a
        variable with `getter`, then modify it in some way.
        This custom getter will create an fp32 variable. If a low precision
        (e.g. float16) variable was requested it will then cast the variable to the
        requested dtype. The reason we do not directly create variables in low
        precision dtypes is that applying small gradients to such variables may
        cause the variable not to change.

        :param getter: The underlying variable getter, that has the same signature as
            tf.get_variable and returns a variable.
        :param name: The name of the variable to get.
        :param shape: The shape of the variable to get.
        :param dtype: The dtype of the variable to get. Note that if this is a low
            precision dtype, the variable will be created as a tf.float32 variable,
            then cast to the appropriate dtype
        :param *args: Additional arguments to pass unmodified to getter.
        :param **kwargs: Additional keyword arguments to pass unmodified to getter.
        :returns
          A variable which is cast to fp16 if necessary.
        """
        if dtype in CASTABLE_TYPES:
            var = getter(name, shape, tf.float32, *args, **kwargs)
            return tf.cast(var, dtype=dtype, name=name + '_cast')
        else:
            return getter(name, shape, dtype, *args, **kwargs)

    def _model_variable_scope(self):
        """Returns a variable scope that the model should be created under.

        if self.dtype is a castable type, model variable will be created in fp32
        then cast to self.dtype before being used.
        :return
        A variable scope for the model.
        """
        return tf.variable_scope('resnet_model', custom_getter=self._custom_dtype_getter)

    def __call__(self, inputs, training):
        with self._model_variable_scope():
            if self.data_format == 'channels_first':
                # Convert the inputs from channels_last(NHWC) to channels_first(NCHW).
                inputs = tf.transpose(a=inputs, perm=[0, 3, 1, 2])

            inputs = conv2d_fixed_padding(
                inputs=inputs, filters=self.num_filters, kernel_size=self.kernel_size,
                strides=self.conv_stride, data_format=self.data_format)

            inputs = tf.identity(inputs, 'initial_conv')

            if self.first_pool_size:
                inputs = tf.layers.max_pooling2d(
                    inputs=inputs, pool_size=self.first_pool_size,
                    strides=self.first_pool_size, padding='SAME',
                    data_format=self.data_format)
                inputs = tf.identity(inputs, 'initial_max_pool')

            for i, num_blocks in enumerate(self.block_sizes):
                num_filters = self.num_filters * (2 ** i)
                inputs = block_layer(
                    inputs=inputs, filters=num_filters, bottleneck=self.bottleneck,
                    block_fn=self.block_fn, blocks = num_blocks,
                    strides=self.block_strides[i], training=training,
                    name='block_layer{}'.format(i + 1), data_format=self.data_format)

            if self.pre_activation:
                inputs = batch_norm(inputs, training, self.data_format)
                inputs = tf.nn.relu(inputs)
            # The current top layer has shape
            # `batch_size x pool_size x pool_size x final_size`.
            # ResNet does an Average Pooling layer over pool_size,
            # but that is the same as doing a reduce_mean. We do a reduce_mean
            # here because it performs better than AveragePooling2D.
            axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]
            inputs = tf.reduce_mean(input_tensor=inputs, axis=axes, keepdims=True)
            inputs = tf.identity(inputs, 'final_reduce_mean')

            inputs = tf.squeeze(inputs, axes)
            inputs = tf.layers.dense(inputs=inputs, units=self.num_classes)
            inputs = tf.identity(inputs, 'final_dense')
            return inputs


if __name__ == '__main__':
    m = Model()
