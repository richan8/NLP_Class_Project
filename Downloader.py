"""NLP_Project.ipynb
Using the [psaw library](https://github.com/dmarx/psaw) to fetch comments from [pushshift.io api](https://github.com/pushshift/api)
"""
#!pip install psaw

import pandas as pd
import datetime as dt
from tqdm import tqdm
from psaw import PushshiftAPI

class Downloader:
  save_local = True
  max_comment_count = 100000
  filter_fields=["body","score","subreddit"]
  api = None
  
  def __init__(self, save_local = True, max_comment_count = 100000):
    self.save_local = save_local
    self.max_comment_count = max_comment_count
    self.api = PushshiftAPI()

  """
  grabs the subreddit and returns the comments as a dataframe
  
  TODO: update code to allow for before and after range
  """
  def fetch_subreddit(self, subreddit_name):
    before_epoch=int(dt.datetime(2020, 10, 16).timestamp())
    comments = []
    
    # rate limit = 1 per second, default
    # comment limit = 500 per requestion, default
    gen = self.api.search_comments(
        subreddit=subreddit_name,
        filter=self.filter_fields,
        before=before_epoch
    )

    with tqdm(total=self.max_comment_count) as pbar:
      for c in gen:
        comments.append(c)
        pbar.update(1)
        if len(comments) >= self.max_comment_count:
          break

    df_subreddit = pd.DataFrame([thing.d_ for thing in comments]).drop(["created_utc", "created"], axis=1)
    df_subreddit = df_subreddit[~(df_subreddit.body.isin(["[removed]","[deleted]"])) & (df_subreddit.score > 0)]

    """Pickle the data for download if necessary"""
    if self.save_local:
      df_subreddit.to_pickle(subreddit_name+".pkl")
    return df_subreddit

  """static loader of pickled subreddit"""
  def load_pickled_sub(subreddit_path):
    return pd.read_pickle(subreddit_path)

