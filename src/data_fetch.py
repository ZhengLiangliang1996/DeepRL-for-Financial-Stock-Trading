
from data.config import DOW_30_TICKER, INDICATORS
from data.data_processor import DataProcessor


TRAIN_START_DATE = '2023-01-01'
TRAIN_END_DATE = '2021-10-01'
TEST_START_DATE = '2021-10-01'
TEST_END_DATE = '2023-01-05'

data_processor = DataProcessor("yahoofinance")
df = data_processor.download_data(start_date = TRAIN_START_DATE,
                                  end_date = TEST_END_DATE,
                                  ticker_list = ['AAPL'],
                                  time_interval='1D')

df = data_processor.preprocess_data(df, technical_list=INDICATORS)


