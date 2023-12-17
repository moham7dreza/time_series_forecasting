from pandora.DataLoader import DataLoader
from pandora.Helper import Helper
from pandora.Multivariate import Multivariate
from pandora.Univariate import Univariate


class Runner:
    @staticmethod
    def run_for_univariate_series_ir(dataset, col, n_steps, epochs):
        dataset = dataset[col].values

        train, test, u_last = DataLoader.train_test_split(dataset, n_steps)

        u_cnn = Univariate.univariant_CNN(train, test, n_steps, epochs)
        u_lstm = Univariate.univariant_LSTM(train, test, n_steps, epochs)
        u_b_lstm = Univariate.univariant_BLSTM(train, test, n_steps, epochs)

        return u_cnn[0], u_lstm[0], u_b_lstm[0], u_last

    @staticmethod
    def run_for_multivariate_series_ir(datasets, col, n_steps, epochs):
        dataset = DataLoader.stack_datasets(datasets, col)

        train, test, last = DataLoader.train_test_split(dataset, n_steps)

        cnn = Multivariate.multivariate_parallel_series_CNN(train, test, n_steps, epochs)
        b_lstm = Multivariate.multivariate_parallel_series_BLSTM(train, test, n_steps, epochs)
        s_lstm = Multivariate.multivariate_parallel_series_Stacked_LSTM(train, test, n_steps, epochs)
        mo_cnn = Multivariate.multivariate_parallel_series_CNN_multiple_output(train, test, n_steps, epochs)

        return cnn, b_lstm, s_lstm, mo_cnn, last
