"""
Usage: hmm.py --crypto_pair=<crypto_pair>

eg. hmm.py --crypto_pair=BTCUSDT
"""
import warnings
import logging
import itertools
import pandas as pd
import numpy as np
import mpl_finance
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from tqdm import tqdm
from docopt import docopt
import binance_api as bn
from datetime import datetime
from datetime import timedelta

args = docopt(doc=__doc__, argv=None, help=True,
              version=None, options_first=False)
warnings.filterwarnings("ignore")
plt.style.use('ggplot')


def mse(result, expected):
    """
    Calculate the mean squared error between the sequences
    result and expected.
    """
    sum_squares = 0
    for num1, num2 in zip(result, expected):
        sum_squares += (num1 - num2) ** 2
    err = float(sum_squares) / len(expected)
    return err


class CryptoPredictor(object):
    def __init__(self, crypto_pair, start_test_date, end_test_date,
                 n_hidden_states=4, n_latency_days=10,
                 n_steps_frac_change=50, n_steps_frac_high=10,
                 n_steps_frac_low=10):

        self._init_logger()
        self._start_test_date = start_test_date
        self._end_test_date = end_test_date
        self.crypto_pair = crypto_pair
        self.n_latency_days = n_latency_days

        self.hmm = GaussianHMM(n_components=n_hidden_states)

        self._split_train_test_data()

        self._compute_all_possible_outcomes(
            n_steps_frac_change, n_steps_frac_high, n_steps_frac_low)

    def _init_logger(self):
        self._logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.DEBUG)

    def _split_train_test_data(self):
        self._logger.info('>>> Fetching historical data')
        self._train_data = bn.get_historical_data('{crypto_pair}'.format(crypto_pair=self.crypto_pair),
                                                  '1 Jun, 2017', '1 Aug, 2018')
        self._test_data = bn.get_test_data('{crypto_pair}'.format(crypto_pair=self.crypto_pair),
                                           self._start_test_date, self._end_test_date)

    @staticmethod
    def _extract_features(data):
        open_price = np.array([float(x[1]) for x in data])
        close_price = np.array([float(x[4]) for x in data])
        high_price = np.array([float(x[2]) for x in data])
        low_price = np.array([float(x[3]) for x in data])

        frac_change = (close_price - open_price) / open_price
        frac_high = (high_price - open_price) / open_price
        frac_low = (open_price - low_price) / open_price

        return np.column_stack((frac_change, frac_high, frac_low))

    def fit(self):
        feature_vector = CryptoPredictor._extract_features(self._train_data)
        self._logger.info('Features extraction completed <<<')

        self.hmm.fit(feature_vector)
        self._logger.info('Data extracted <<<')

    def _compute_all_possible_outcomes(self, n_steps_frac_change,
                                       n_steps_frac_high, n_steps_frac_low):
        frac_change_range = np.linspace(-0.1, 0.1, n_steps_frac_change)
        frac_high_range = np.linspace(0, 0.1, n_steps_frac_high)
        frac_low_range = np.linspace(0, 0.1, n_steps_frac_low)

        self._possible_outcomes = np.array(list(itertools.product(
            frac_change_range, frac_high_range, frac_low_range)))

    def _get_most_probable_outcome(self, day_index):
        previous_data_start_index = max(0, day_index - self.n_latency_days)
        previous_data_end_index = max(0, day_index - 1)
        previous_data = self._test_data[previous_data_end_index: previous_data_start_index]
        previous_data_features = CryptoPredictor._extract_features(previous_data)

        outcome_score = []
        for possible_outcome in self._possible_outcomes:
            total_data = np.row_stack((previous_data_features, possible_outcome))
            outcome_score.append(self.hmm.score(total_data))
        most_probable_outcome = self._possible_outcomes[np.argmax(outcome_score)]

        return most_probable_outcome

    def predict_close_price(self, day_index):
        open_price = (float(self._test_data[day_index][1]) + float(self._test_data[day_index][4]))/2
        predicted_frac_change, _, _ = self._get_most_probable_outcome(day_index)
        return open_price * (1 + predicted_frac_change)

    def predict_close_prices_for_days(self, days, with_plot=False):
        predicted_close_prices = []

        for day_index in tqdm(range(days)):
            predicted_close_prices.append(self.predict_close_price(day_index))
        predicted_close_prices.pop(0)
        test_data = [float(x[4]) for x in self._test_data[: days]]
        actual_close_prices = test_data
        actual_close_prices.pop()
        err = mse(predicted_close_prices, actual_close_prices)
        if with_plot:

            days_actual = [datetime.strptime(self._start_test_date, '%d %b, %Y') + timedelta(days=x) for x in range(0, days-1)]
            days_predict = [datetime.strptime(self._start_test_date, '%d %b, %Y') + timedelta(days=x) for x in range(0, days-1)]

            fig = plt.figure()

            axes = fig.add_subplot(111)
            axes.plot(days_actual, actual_close_prices, 'bo-', label="actual")
            axes.plot(days_predict, predicted_close_prices, 'r+-', label="predicted")
            axes.set_title('{crypto_pair}'.format(crypto_pair=self.crypto_pair))

            fig.autofmt_xdate()
            plt.legend()
            plt.show()

        return err


stock_predictor = CryptoPredictor(crypto_pair=args['--crypto_pair'], start_test_date='2 Aug, 2018', end_test_date='11 Jan, 2019')
stock_predictor.fit()
print(stock_predictor.predict_close_prices_for_days(50, with_plot=True))