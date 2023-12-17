from pandora.Config import Config
from pandora.DataSampler import DataSampler
from pandora.ModelBuilder import ModelBuilder
from pandora.Helper import Helper


class Univariate:
    epochs = Config.epochs_for_univariate_series
    @staticmethod
    def univariant_CNN(dataset, x_input, n_steps, epochs):
        # split into samples
        X, y = DataSampler.split_sequence_for_univariate_series(dataset, n_steps)
        # reshape from [samples, timesteps] into [samples, timesteps, features]
        n_features = 1
        X = X.reshape((X.shape[0], X.shape[1], n_features))
        # define model
        model = ModelBuilder.get_CNN_model(n_steps, n_features)
        # fit model
        model.fit(X, y, epochs, verbose=0)
        # demonstrate prediction

        x_input = x_input.reshape((1, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)
        yhat = Helper.flatten_arr(yhat)
        print(yhat)
        return yhat
    @staticmethod
    def univariant_LSTM(dataset, x_input, n_steps, epochs):
        # split into samples
        X, y = DataSampler.split_sequence_for_univariate_series(dataset, n_steps)
        # reshape from [samples, timesteps] into [samples, timesteps, features]
        n_features = 1
        X = X.reshape((X.shape[0], X.shape[1], n_features))
        # define model
        model = ModelBuilder.get_stacked_LSTM_model(n_steps, n_features)
        # fit model
        model.fit(X, y, epochs, verbose=0)
        # demonstrate prediction

        x_input = x_input.reshape((1, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)
        yhat = Helper.flatten_arr(yhat)
        print(yhat)
        return yhat
    @staticmethod
    def univariant_BLSTM(dataset, x_input, n_steps, epochs):
        # split into samples
        X, y = DataSampler.split_sequence_for_univariate_series(dataset, n_steps)
        # reshape from [samples, timesteps] into [samples, timesteps, features]
        n_features = 1
        X = X.reshape((X.shape[0], X.shape[1], n_features))
        # define model
        model = ModelBuilder.get_bidirectional_LSTM_model(n_steps, n_features)
        # fit model
        model.fit(X, y, epochs, verbose=0)
        # demonstrate prediction

        x_input = x_input.reshape((1, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)
        yhat = Helper.flatten_arr(yhat)
        print(yhat)
        return yhat