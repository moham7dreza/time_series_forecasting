from keras.layers import Dense, Flatten, Input, Conv1D, MaxPooling1D, LSTM, Bidirectional
from keras.models import Sequential, Model


class ModelBuilder:
    @staticmethod
    def get_CNN_model(n_steps, n_features):
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(n_features))
        model.compile(optimizer='adam', loss='mse')
        return model

    @staticmethod
    def get_stacked_LSTM_model(n_steps, n_features):
        model = Sequential()
        model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
        model.add(LSTM(100, activation='relu'))
        model.add(Dense(n_features))
        model.compile(optimizer='adam', loss='mse')
        return model

    @staticmethod
    def get_multi_output_CNN_model(n_steps, n_features):
        # define model
        visible = Input(shape=(n_steps, n_features))
        cnn = Conv1D(filters=64, kernel_size=2, activation='relu')(visible)
        cnn = MaxPooling1D(pool_size=2)(cnn)
        cnn = Flatten()(cnn)
        cnn = Dense(50, activation='relu')(cnn)
        # define outputs dynamically based on n_features
        outputs = [Dense(1)(cnn) for _ in range(n_features)]
        # tie together
        model = Model(inputs=visible, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')
        return model

    @staticmethod
    def get_bidirectional_LSTM_model(n_steps, n_features):
        # define model
        model = Sequential()
        model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))
        model.add(Dense(n_features))
        model.compile(optimizer='adam', loss='mse')
        return model
