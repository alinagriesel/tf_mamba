# imports
import tensorflow as tf
from einops import rearrange, repeat

class MambaResBlock(tf.keras.Model):

  def __init__(self, input_dim):
    super().__init__()

    self.input_dim = input_dim
    self.projection_dim = 2*input_dim

    # normalisation
    self.layernorm = tf.keras.layers.LayerNormalization()
    # Dense
    self.dense1 = tf.keras.layers.Dense(units=self.projection_dim)
    # Dense
    self.dense2 = tf.keras.layers.Dense(units=self.projection_dim)
    # Convolution
    self.conv1d = tf.keras.layers.Conv1D(filters=self.projection_dim , kernel_size=4, strides=1 , padding="causal", groups = self.projection_dim, data_format = "channels_last") # data_format?, groups?
    # SSM block
    self.ssm = SelectiveSSM(32, self.projection_dim)
    # Dense
    self.dense3 = tf.keras.layers.Dense(units=input_dim)
    # dropout
    self.dropout = tf.keras.layers.Dropout(rate=0.2)

  # forward step
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
  def __init__(self, states, projection_dim):
    super().__init__()

    self.states = states
    self.projection_dim = projection_dim

    # hippo initialisation für A ? dafür müsste A aber quadratisch sein
    # -> quadratische matrix oder nicht ?
    #self.A =  # states x internal dim
    #self.D =  # np ones internal dim
    A = repeat(tf.range(1, states+1, dtype=tf.float32),'n -> d n', d=self.projection_dim)

    # A parameter: matrix changing hidden state
    self.A_log = tf.Variable(tf.math.log(A),trainable=True, dtype=tf.float32)

    # D parameter: simple skip connection, initialisde as ones
    self.D = tf.Variable(tf.ones(projection_dim),trainable=True, dtype=tf.float32) # change from np to tf

    # B, C, delta parameter: dependend on input, therefore dense layer to learn these
    self.denseB = tf.keras.layers.Dense(units=self.states)
    self.denseC = tf.keras.layers.Dense(units=self.states)
    self.densedelta = tf.keras.layers.Dense(units=self.projection_dim)

  def selective_scan(self,input, delta, A, B, C, D):
    """
    Calculate output of the selective state space model using parallel scan
    implemented using the cumulative sum

    args:
      input: data input that we calculate the ssm on
      delta: mediates how much focus is put on new input
      A: state matrix controlling the hidden state
      B: modulate the recurrent dynamics based on content (input)
      C: modulate the recurrent dynamics based on context (hidden states)
      D: scales the skip connection

    returns:
      output: result of the ssm with current parameters

    """
    # first step of discretization of A
    deltaA = tf.einsum('bld,dn->bldn', delta, A) # quasi delta mal A
    deltaBinput = tf.einsum('bld,bld,bln->bldn', delta, input, B) # input mal B mal delta

    deltaA_cumsum = tf.pad(
        deltaA[:, 1:], [[0, 0], [1, 1], [0, 0], [0, 0]])[:, 1:, :, :]

    deltaA_cumsum = tf.reverse(deltaA_cumsum, axis=[1])  # Flip along axis 1

    # Cumulative sum along all the input tokens, parallel prefix sum,
    # calculates dA for all the input tokens in parallel
    deltaA_cumsum = tf.math.cumsum(deltaA_cumsum, axis=1)

    # second step of discretization of A
    deltaA_cumsum = tf.exp(deltaA_cumsum)
    deltaA_cumsum = tf.reverse(deltaA_cumsum, axis=[1])  # Flip back along axis 1

    # calculate intermediate output as in graphs shown for ssm's
    x = deltaBinput * deltaA_cumsum
    # 1e-12 to avoid division by 0
    x = tf.math.cumsum(x, axis=1)/(deltaA_cumsum + 1e-12)

    # intermediate output multiplied with parameter C
    output = tf.einsum('bldn,bln->bld', x, C)

    return output + input * D

  def call(self, input):

    A = -tf.exp(tf.cast(self.A_log, tf.float32))

    C = self.denseC(input)
    B = self.denseB(input)
    # softplus to not get nan values
    delta = tf.nn.softplus(self.densedelta(input))

    return self.selective_scan(input, delta, A, B, C, self.D)

class MambaModel(tf.keras.Model):
  def __init__(self, num_layers, vocab_size, input_dim):
    super().__init__()

    self.num_layers = num_layers

    self.embedding = tf.keras.layers.Embedding(input_dim = vocab_size, output_dim = input_dim) #, input_length = 128) (bs, 128, 128)
    self.layer_list = []
    for i in range(num_layers):
        self.layer_list.append(MambaResBlock(input_dim))
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.flatten = tf.keras.layers.Flatten()
    self.dense = tf.keras.layers.Dense(units=1024, activation=tf.nn.gelu)
    self.out = tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid)

  def call(self, input):

    x = self.embedding(input)

    for i in range(self.num_layers):
      x = self.layer_list[i](x)

    x = self.flatten(x)
    x = self.dense(x)
    x = self.out(x)

    return x