import pandas as pd
import numpy as np
from binance.client import Client
import snscrape.modules.twitter as sntwitter
from sqlalchemy import create_engine

import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from stop_words import get_stop_words


def con_database():
    user = ""
    password = ""
    host = ""
    database = ""

    sqlEngine = create_engine(
        'mysql://{user}:{pwd}@{host}/{db}'.format(user=user, pwd=password, host=host, db=database), echo=False)
    return sqlEngine


def con_binance():
    api_key = ""
    api_secret = ""
    return Client(api_key, api_secret)


def get_tweets(engine):
    last_tweet_id = int(pd.read_sql('SELECT max(id) FROM dbtwitter.tweet;', con=engine).iloc[0, 0])

    # Creating list to append tweet data to
    tweets_list = []

    # Using TwitterSearchScraper to scrape data and append tweets to list
    for tweet in sntwitter.TwitterSearchScraper('bitcoin lang:en -filter:replies').get_items():
        if tweet.id <= last_tweet_id:
            break
        tweets_list.append([tweet.id, tweet.date, tweet.content])

    # Creating a dataframe from the tweets list above
    df = pd.DataFrame(tweets_list, columns=['id', 'created_at', 'tweet'])
    df = df.set_index('id')
    return df


# cleaning the tweets
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt


def replace_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, ' ', input_txt)
    return input_txt


def clean_tweets(tweets):
    # remove twitter Return handles (RT @xxx:)
    tweets = np.vectorize(remove_pattern)(tweets, "RT @[\w]*:")

    # remove twitter handles (@xxx)
    tweets = np.vectorize(remove_pattern)(tweets, "@[\w]*")

    # remove URL links (httpxxx)
    tweets = np.vectorize(remove_pattern)(tweets, "https?://[A-Za-z0-9./]*")

    # replace (#)
    tweets = np.vectorize(replace_pattern)(tweets, "#")
    return tweets


def remove_stopWords(s, stp):
    s = ' '.join(word for word in s.split() if word not in stp)
    return s


def analysis(df):
    stop_words = get_stop_words('en')
    analyzer = SentimentIntensityAnalyzer()

    df['tweet'] = clean_tweets(df['tweet'])
    score = [analyzer.polarity_scores(tweet)['compound'] for tweet in df['tweet']]
    df['score'] = score

    df.loc[:, "tweet"] = df.tweet.apply(lambda x: str.lower(x))
    df.loc[:, "tweet"] = df.tweet.apply(lambda x: str.lower(x))
    df.loc[:, "tweet"] = df.tweet.apply(lambda x: " ".join(re.findall('[\w]+', x)))
    df.loc[:, "tweet"] = df.tweet.apply(lambda x: remove_stopWords(x, stop_words))
    df.loc[:, "tweet"] = df.tweet.apply(lambda x: x.encode('ascii', 'ignore').decode('ascii'))
    return df


def get_data_crypto(engine, bin_client, crypto):
    last_datetime = str(
        pd.read_sql('SELECT max(dateTime) FROM dbtwitter.binance WHERE crypto = "{}";'.format(crypto), con=engine).iloc[
            0, 0])
    candles = bin_client.get_historical_klines(crypto, Client.KLINE_INTERVAL_1MINUTE, last_datetime)

    df = pd.DataFrame(candles,
                      columns=['dateTime', 'open', 'high', 'low', 'close', 'volume', 'closeTime', 'quoteAssetVolume',
                               'numberOfTrades', 'takerBuyBaseVol', 'takerBuyQuoteVol', 'ignore'])
    df = df[
        ['dateTime', 'open', 'high', 'low', 'close', 'volume', 'quoteAssetVolume', 'numberOfTrades', 'takerBuyBaseVol',
         'takerBuyQuoteVol']]
    df.dateTime = pd.to_datetime(df.dateTime, unit='ms')

    colnames_to_float = ['open', 'high', 'low', 'close', 'volume', 'quoteAssetVolume', 'takerBuyBaseVol',
                         'takerBuyQuoteVol']
    df[colnames_to_float] = df[colnames_to_float].astype(float, errors='raise')

    df = df[df['dateTime'] != df['dateTime'].min()]
    df = df[df['dateTime'] != df['dateTime'].max()]
    df['crypto'] = crypto
    return df


def send_to_mysql(df, sqlEngine, name, index_tf):
    df.to_sql(name, con=sqlEngine, if_exists='append', index=index_tf)


def final(a, b):
    engine = con_database()
    binance_client = con_binance()
    df_tweet = get_tweets(engine)
    df_clean = analysis(df_tweet)
    df_bitcoin = get_data_crypto(engine, binance_client, "BTCUSDT")
    send_to_mysql(df_clean, engine, 'tweet', True)
    send_to_mysql(df_bitcoin, engine, 'binance', False)
