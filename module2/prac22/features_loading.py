# coding: utf-8
import pandas as pd
from sqlalchemy import create_engine
import os
import pickle
from catboost import CatBoostClassifier

engine = create_engine(
    "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
    "postgres.lab.karpov.courses:6432/startml"
)

post_query = "SELECT * FROM public.post_text_df"

post_df = pd.read_sql(post_query, engine)


user_query = "SELECT DISTINCT user_id, age, city, country, exp_group, gender, os, source  FROM public.user_data"

feed_query = "SELECT * FROM public.feed_data LIMIT 200000"

user_df = pd.read_sql(user_query, engine)

feed_df = pd.read_sql(feed_query, engine)

df = pd.merge(user_df,
              feed_df,
              on='user_id',
              how='left')

df = pd.merge(df,
              post_df,
              on='post_id',
              how='left')

df['day'] = pd.to_datetime(df['timestamp']).apply(lambda x: x.day)
df['month'] = pd.to_datetime(df['timestamp']).apply(lambda x: x.month)

df = df.drop([
    'action',
    'text',
],
    axis=1)

df = df.set_index(['user_id', 'post_id'])

df.to_sql('bezyazychnyy_artem_features_lesson_22', con=engine,schema="public",                   
    if_exists='replace')

def load_features()->pd.DataFrame:
    df=batch_load_sql('SELECT * FROM bezyazychnyy_artem_features_lesson_22')
    return df

def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)
