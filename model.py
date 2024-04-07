# imports
import tensorflow as tf
from einops import rearrange, repeat

class MambaResBlock(tf.keras.Model):

  def __init__(self, input_dim, projection_dim):
    super().__init__()

    # normalisation
    self.layernorm = tf.keras.layers.LayerNormalization()
    # Dense
    self.dense1 = tf.keras.layers.Dense(units=projection_dim) # activation ?
    # Dense
    self.dense2 = tf.keras.layers.Dense(units=projection_dim) # activation ?
    # Convolution
    self.conv1d = tf.keras.layers.Conv1D(filters=projection_dim , kernel_size=4, strides=1 , padding="causal", groups = 256, data_format = "channels_last") # data_format?, groups?
    # SSM block
    self.ssm = SelectiveSSM(32, 256)
    # Dense
    self.dense3 = tf.keras.layers.Dense(units=input_dim) # activation ?
    # dropout
    self.dropout = tf.keras.layers.Dropout(rate=0.2)

  def call(self, input):

    x = self.layernorm(input)

    x1 = self.dense1(x)
    x1 = self.conv1d(x1)
    x1 =  tf.nn.silu(x1)
    x1 = self.ssm(x1)

    x2 = self.dense2(x)
    x2 =  tf.nn.silu(x2)

    x = x1 * x2
    x = self.dense3(x)

    # skip connection
    x = x + input

    x = self.dropout(x)

    return x

class SelectiveSSM(tf.keras.Model):
  def __init__(self, states, internal_dim):
    super().__init__()

    self.states = states
    self.internal_dim = internal_dim

    # hippo initialisation für A ? dafür müsste A aber quadratisch sein
    # -> quadratische matrix oder nicht ?
    #self.A =  # states x internal dim
    #self.D =  # np ones internal dim
    A = repeat(tf.range(1, states+1, dtype=tf.float32),'n -> d n', d=internal_dim)

    self.A_log = tf.Variable(tf.math.log(A),trainable=True, dtype=tf.float32)

    self.D = tf.Variable(tf.ones(internal_dim),trainable=True, dtype=tf.float32) # change from np to tf

    self.denseB = tf.keras.layers.Dense(units=self.states)
    self.denseC = tf.keras.layers.Dense(units=self.states)
    self.densedelta = tf.keras.layers.Dense(units=self.internal_dim)

  def selective_scan(self,u, delta, A, B, C, D):
    # first step of A_bar = exp(ΔA), i.e., ΔA
    dA = tf.einsum('bld,dn->bldn', delta, A) # quasi delta mal A
    dB_u = tf.einsum('bld,bld,bln->bldn', delta, u, B) # input mal B mal delta

    dA_cumsum = tf.pad(
        dA[:, 1:], [[0, 0], [1, 1], [0, 0], [0, 0]])[:, 1:, :, :]

    dA_cumsum = tf.reverse(dA_cumsum, axis=[1])  # Flip along axis 1

    # Cumulative sum along all the input tokens, parallel prefix sum,
    # calculates dA for all the input tokens parallely
    dA_cumsum = tf.math.cumsum(dA_cumsum, axis=1)

    # second step of A_bar = exp(ΔA), i.e., exp(ΔA)
    dA_cumsum = tf.exp(dA_cumsum)
    dA_cumsum = tf.reverse(dA_cumsum, axis=[1])  # Flip back along axis 1

    x = dB_u * dA_cumsum
    # 1e-12 to avoid division by 0
    x = tf.math.cumsum(x, axis=1)/(dA_cumsum + 1e-12)

    y = tf.einsum('bldn,bln->bld', x, C)

    return y + u * D

  def call(self, input):

    A = -tf.exp(tf.cast(self.A_log, tf.float32)) # shape -> (d_in, n)
    #D = tf.cast(self.D, tf.float32)

    C = self.denseC(input)
    B = self.denseB(input)
    delta = tf.nn.softplus(self.densedelta(input))

    return self.selective_scan(input, delta, A, B, C, self.D)

class MambaModel(tf.keras.Model):
  def __init__(self, num_layers, vocab_size):
    super().__init__()

    self.num_layers = num_layers

    self.embedding = tf.keras.layers.Embedding(input_dim = vocab_size, output_dim = 128) #, input_length = 128) (bs, 128, 128)
    self.layer_list = []
    for i in range(num_layers):
        self.layer_list.append(MambaResBlock(128, 256))
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.flatten = tf.keras.layers.Flatten()
    self.dense = tf.keras.layers.Dense(units=1024, activation=tf.nn.gelu)
    self.out = tf.keras.layers.Dense(units=vocab_size, activation=tf.nn.softmax)

  def call(self, input):

    x = self.embedding(input)

    for i in range(self.num_layers):
      x = self.layer_list[i](x)

    x = self.flatten(x)
    x = self.dense(x)
    x = self.out(x)

    return x