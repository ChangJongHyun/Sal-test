import keras
from keras.layers import *
from keras.models import Model
import keras.backend as K


def convLSTM(filters, kernal_size, padding='same', return_sequences=True,
             initializer=None, activation=None):
    if initializer is None:
        initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
    if activation is None:
        activation = LeakyReLU(alpha=0.2)
    return ConvLSTM2D(filters=filters,
                      kernel_size=(kernal_size, kernal_size),
                      padding=padding, activation=activation,
                      return_sequences=return_sequences,
                      bias_initializer=initializer,
                      kernel_initializer=initializer,
                      recurrent_initializer=initializer)


def discriminator_loss(y_true, y_pred):
    return None

def construct_network(state_input, action_input):

    state_input = Input(shape=state_input)
    action_input = Input(shape=action_input)

    y = Dense(64, activation='relu')(action_input)

    x = convLSTM(64, 3)(state_input)
    x = BatchNormalization()(x)
    x = convLSTM(64, 3)(x)
    x = BatchNormalization()(x)
    x = convLSTM(64, 3)(x)
    x = BatchNormalization()(x)
    x = convLSTM(64, 3, return_sequences=False)(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)

    z = keras.layers.concatenate([x, y])
    z = Dropout(rate=0.5)(z)
    z = Dense(1, activation='sigmoid')(z)

    model = Model(inputs=[state_input, action_input], outputs=z)
    # model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


def construct_discriminator(expert_obs, agent_obs, expert_action, agent_action):
    state_in = (None, 224, 224, 3)
    action_in = (2,)
    discrim = construct_network(state_in, action_in)

    expert_obs_in = Input(shape=expert_obs, name='expert_obs_in')
    expert_action_in = Input(shape=expert_action, name='expert_action_in')
    expert_in = [expert_obs_in, expert_action_in]

    agent_obs_in = Input(shape=agent_obs, name='agent_obs_in')
    agent_action_in = Input(shape=agent_action, name='agent_action_in')
    agent_in = [agent_obs_in, agent_action_in]

    expert_net = discrim(expert_in)
    agent_net = discrim(agent_in)

    model = Model(inputs=expert_in + agent_in)
    model.compile(loss=keras.losses.binary_crossentropy(expert_net, agent_net),
                  optimizer='adam')
    model.summary()


class Discriminator:
    def __init__(self, obs_in, action_in):
        self.discrim_network = construct_network(obs_in, action_in)

        expert_loss = keras.losses.binary_crossentropy()
        agent_loss = keras.losses.binary_crossentropy()
        self.loss = - (expert_loss + agent_loss)
        optimizer = keras.optimizers.Adam()

    def train(self, memory, demos):
        expert_in = Input()


class Policy:
    def __init__(self, obs_in, action_in):
        self.policy_network = construct_network(obs_in, action_in)
        self.value_network = construct_network(obs_in, action_in)


class GAIL:
    def __init__(self, observation_space, action_space):
        self.policy_net = None
        self.discriminate_nat = None
        self.observation_space = observation_space
        self.action_space = action_space

    # PI(s, a) - actor
    def build_policy(self, state_input, action_input):
        policy_network = self.construct_network(state_input, action_input)

    # Q(s, a) - critic
    def build_value(self, state_input, action_input):
        value_network = self.construct_network(state_input, action_input)

    # D(s,a)
    def build_discriminator(self, state_input, action_input):
        agent = 'agent_'
        expert = 'expert_'

        x = convLSTM(64, 3)(state_input)
        x = BatchNormalization()(x)

        x = convLSTM(64, 3)(x)
        x = BatchNormalization()(x)

        x = convLSTM(64, 3)(x)
        x = BatchNormalization()(x)

        x = convLSTM(64, 3)(x)
        x = BatchNormalization()(x)

        x = Conv3D(filters=1, kernel_size=(3, 3, 3),
                   activation=LeakyReLU(alpha=0.2),
                   padding='same', data_format='channels_last')(x)
        x = Flatten()(x)

        y = Dense(64, activation='relu')(action_input)

        z = keras.layers.Add()([x, y])
        z = Dropout(rate=0.5)(z)
        z = Dense(1, activation='sigmoid')(z)

        model = Model(inputs=[state_input, action_input], outputs=z)
        model.compile(loss='binary_crossentropy', optimizer='adam')
        expert_network = self.construct_network(state_input, action_input)
        agent_network = self.construct_network(state_input, action_input)


if __name__ == '__main__':
    state_size = (None, 224, 224, 3)
    action_size = (2,)
    construct_discriminator(state_size, state_size, action_size, action_size)

