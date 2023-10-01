from tensorflow.keras import models, layers
from tensorflow.keras import regularizers
import tensorflow as tf
from tensorflow import keras

def UNet(input_shape,input_label_channel, layer_count=64, regularizers = regularizers.l2(0.0001), gaussian_noise=0.1, weight_file = None):
        """ Method to declare the UNet model.

        Args:
            input_shape: tuple(int, int, int, int)
                Shape of the input in the format (batch, height, width, channels).
            input_label_channel: list([int])
                list of index of label channels, used for calculating the number of channels in model output.
            layer_count: (int, optional)
                Count of kernels in first layer. Number of kernels in other layers grows with a fixed factor.
            regularizers: keras.regularizers
                regularizers to use in each layer.
            weight_file: str
                path to the weight file.
        """

        input_img = layers.Input(input_shape[1:], name='Input')
        pp_in_layer  = input_img
#        pp_in_layer = layers.GaussianNoise(gaussian_noise)(input_img)
#        pp_in_layer = layers.BatchNormalization()(pp_in_layer)


        c1 = layers.Conv2D(1*layer_count, (3, 3), activation='relu', padding='same')(pp_in_layer)
        c1 = layers.Conv2D(1*layer_count, (3, 3), activation='relu', padding='same')(c1)
        n1 = layers.BatchNormalization()(c1)
        p1 = layers.MaxPooling2D((2, 2))(n1)
        p1 = CBAM_attention(p1) + p1

        c2 = layers.Conv2D(2*layer_count, (3, 3), activation='relu', padding='same')(p1)
        c2 = layers.Conv2D(2*layer_count, (3, 3), activation='relu', padding='same')(c2)
        n2 = layers.BatchNormalization()(c2)
        p2 = layers.MaxPooling2D((2, 2))(n2)
        p2 = CBAM_attention(p2) + p2

        c3 = layers.Conv2D(4*layer_count, (3, 3), activation='relu', padding='same')(p2)
        c3 = layers.Conv2D(4*layer_count, (3, 3), activation='relu', padding='same')(c3)
        n3 = layers.BatchNormalization()(c3)
        p3 = layers.MaxPooling2D((2, 2))(n3)
        p3 = CBAM_attention(p3) + p3

        c4 = layers.Conv2D(8*layer_count, (3, 3), activation='relu', padding='same')(p3)
        c4 = layers.Conv2D(8*layer_count, (3, 3), activation='relu', padding='same')(c4)
        n4 = layers.BatchNormalization()(c4)
        p4 = layers.MaxPooling2D(pool_size=(2, 2))(n4)
        p4 = CBAM_attention(p4) + p4

        c5 = layers.Conv2D(16*layer_count, (3, 3), activation='relu', padding='same')(p4)
        c5 = layers.Conv2D(16*layer_count, (3, 3), activation='relu', padding='same')(c5)

        u6 = layers.UpSampling2D((2, 2))(c5)
        n6 = layers.BatchNormalization()(u6)
        u6 = layers.concatenate([n6, n4])
        c6 = layers.Conv2D(8*layer_count, (3, 3), activation='relu', padding='same')(u6)
        c6 = layers.Conv2D(8*layer_count, (3, 3), activation='relu', padding='same')(c6)

        u7 = layers.UpSampling2D((2, 2))(c6)
        n7 = layers.BatchNormalization()(u7)
        u7 = layers.concatenate([n7, n3])
        c7 = layers.Conv2D(4*layer_count, (3, 3), activation='relu', padding='same')(u7)
        c7 = layers.Conv2D(4*layer_count, (3, 3), activation='relu', padding='same')(c7)

        u8 = layers.UpSampling2D((2, 2))(c7)
        n8 = layers.BatchNormalization()(u8)
        u8 = layers.concatenate([n8, n2])
        c8 = layers.Conv2D(2*layer_count, (3, 3), activation='relu', padding='same')(u8)
        c8 = layers.Conv2D(2*layer_count, (3, 3), activation='relu', padding='same')(c8)

        u9 = layers.UpSampling2D((2, 2))(c8)
        n9 = layers.BatchNormalization()(u9)
        u9 = layers.concatenate([n9, n1])
        c9 = layers.Conv2D(1*layer_count, (3, 3), activation='relu', padding='same')(u9)
        c9 = layers.Conv2D(1*layer_count, (3, 3), activation='relu', padding='same')(c9)
        c9 = layers.Dropout(.2)(c9)#, training=True)
        
        d = layers.Conv2D(len(input_label_channel), (1, 1), activation='sigmoid', kernel_regularizer= regularizers)(c9)

        seg_model = models.Model(inputs=[input_img], outputs=[d])
        if weight_file:
            seg_model.load_weights(weight_file)
        seg_model.summary()
        return seg_model

# (1) Channel Attention
def channel_attention(inputs, ratio=0.25):
    '''ratio represents the multiplier for reducing the number of channels in the first fully connected layer'''

    channel = inputs.shape[-1]  # Get the number of channels in the input feature map

    # Apply global max-pooling and global average-pooling to the output feature map separately
    # [h,w,c] => [None,c]
    x_max = layers.GlobalMaxPooling2D()(inputs)
    x_avg = layers.GlobalAveragePooling2D()(inputs)

    # [None,c] => [1,1,c]
    x_max = layers.Reshape([1,1,-1])(x_max)  # -1 automatically finds the channel dimension size
    x_avg = layers.Reshape([1,1,-1])(x_avg)  # Alternatively, you can use the variable 'channel' instead of -1

    # Reduce the number of channels by 1/4 in the first fully connected layer, [1,1,c] => [1,1,c//4]
    x_max = layers.Dense(channel*ratio)(x_max)
    x_avg = layers.Dense(channel*ratio)(x_avg)

    # Apply ReLU activation
    x_max = layers.Activation('relu')(x_max)
    x_avg = layers.Activation('relu')(x_avg)

    # Increase the number of channels in the second fully connected layer, [1,1,c//4] => [1,1,c]
    x_max = layers.Dense(channel)(x_max)
    x_avg = layers.Dense(channel)(x_avg)

    # Sum the results, [1,1,c] + [1,1,c] => [1,1,c]
    x = layers.Add()([x_max, x_avg])

    # Normalize the weights using sigmoid
    x = tf.nn.sigmoid(x)

    # Multiply the input feature map by the weight vector to assign weights to each channel
    x = layers.Multiply()([inputs, x])  # [h,w,c] * [1,1,c] => [h,w,c]

    return x

# (2) Spatial Attention
def spatial_attention(inputs):

    # Perform max-pooling and average-pooling over the channel dimension [b,h,w,c] => [b,h,w,1]
    # Set keepdims=False to get [b,h,w,c] => [b,h,w]
    x_max = tf.reduce_max(inputs, axis=3, keepdims=True)  # Compute the maximum value over the channel dimension
    x_avg = tf.reduce_mean(inputs, axis=3, keepdims=True)  # 'axis' can also be -1

    # Stack the results over the channel dimension [b,h,w,2]
    x = layers.concatenate([x_max, x_avg])

    # Adjust the channels using a 1*1 convolution [b,h,w,1]
    x = layers.Conv2D(filters=1, kernel_size=(1,1), strides=1, padding='same')(x)

    # Normalize the weights using the sigmoid function
    x = tf.nn.sigmoid(x)

    # Multiply the input feature map by the weight vector
    x = layers.Multiply()([inputs, x])

    return x

# (3) CBAM Attention
def CBAM_attention(inputs):

    # Apply channel attention first and then spatial attention
    x = channel_attention(inputs)
    x = spatial_attention(x)
    return x