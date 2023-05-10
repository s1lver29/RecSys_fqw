import random
import warnings
import sys

from tqdm.notebook import trange
from loguru import logger

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

warnings.filterwarnings("ignore")


class UserItemRatingDataset(Dataset):
    def __init__(self, user: list, item: list, rating: list):
        super(UserItemRatingDataset, self).__init__()

        self.user = torch.tensor(user, dtype=torch.long)
        self.item = torch.tensor(item, dtype=torch.long)
        self.target = torch.tensor(rating, dtype=torch.long)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.user[idx], self.item[idx], self.target[idx]


class NCFData(object):
    def __init__(self, ratings, num_negatives, num_negatives_test, batch_size: int):
        self.ratings = ratings
        self.num_negatives = num_negatives
        self.num_negatives_test = num_negatives_test
        self.batch_size = batch_size

        self.preprocess_ratings = self._reindex(self.ratings)
        self.user_pool = set(self.ratings["user_id"].unique())
        self.item_pool = set(self.ratings["item_id"].unique())

        logger.info("Start leave one out")
        self.train_ratings, self.test_ratings = self._leave_one_out(
            self.preprocess_ratings
        )
        logger.info("Start negative sampling")
        self.negatives = self._negative_sampling(self.preprocess_ratings)

    def _reindex(self, ratings):
        logger.info("Start reindex")
        user = list(ratings["user_id"].drop_duplicates())
        logger.info("user2id")
        self.user2id = {w: i for i, w in enumerate(user)}

        logger.info("item2id start")
        item = list(ratings["item_id"].drop_duplicates())
        self.item2id = {w: i for i, w in enumerate(item)}

        ratings["user_id"] = ratings["user_id"].apply(lambda x: self.user2id[x])
        ratings["item_id"] = ratings["item_id"].apply(lambda x: self.item2id[x])
        ratings["rating"] = ratings["rating"].apply(lambda x: float(x > 0))

        logger.info("Return ratings")
        return ratings

    def _leave_one_out(self, ratings):
        ratings["rank_latest"] = ratings.groupby(["user_id"])["timestamp"].rank(
            method="first", ascending=True
        )
        test = ratings.loc[ratings["rank_latest"] == 1]
        train = ratings.loc[ratings["rank_latest"] > 1]
        return (
            train[["user_id", "item_id", "rating"]],
            test[["user_id", "item_id", "rating"]],
        )

    def _negative_sampling(self, ratings):
        interact_status = (
            ratings.groupby("user_id")["item_id"]
            .apply(set)
            .reset_index()
            .rename(columns={"item_id": "interacted_items"})
        )
        # logger.info("Negative items")
        # interact_status["negative_items"] = [
        #     self.item_pool - ratings["user_id"].iloc[i]
        #     for i in ratings["user_id"].shape[0]
        # ]

        logger.info("Negative samples")
        interact_status["negative_samples"] = [
            random.sample(
                self.item_pool - interact_status["interacted_items"].iloc[i],
                self.num_negatives_test,
            )
            for i in range(interact_status["interacted_items"].shape[0])
        ]

        logger.info("Return interact status")
        return interact_status[["user_id", "negative_items", "negative_samples"]]

    def get_train_instance(self):
        users, items, ratings = [], [], []
        train_ratings = pd.merge(
            self.train_ratings,
            self.negatives[["user_id", "negative_items"]],
            on="user_id",
        )
        train_ratings["negatives"] = train_ratings["negative_items"].apply(
            lambda x: random.sample(x, self.num_negatives)
        )

        for row in train_ratings.itertuples():
            users.append(int(row.user_id))
            items.append(int(row.item_id))
            ratings.append(float(row.rating))

        dataset = UserItemRatingDataset(user=users, item=items, rating=ratings)
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def get_test_instance(self):
        users, items, ratings = [], [], []
        test_ratings = pd.merge(
            self.test_ratings,
            self.negatives[["user_id", "negative_items"]],
            on="user_id",
        )

        for row in test_ratings.itertuples():
            users.append(int(row.user_id))
            items.append(int(row.item_id))
            ratings.append(float(row.rating))

        dataset = UserItemRatingDataset(user=users, item=items, rating=ratings)
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )


if __name__ == "__main__":
    logger.remove()
    logger.add(
        sys.__stdout__,
        format="[{time:YYYY-MM-DD HH:mm:ss}] {level} | {message}",
        level="TRACE",
        colorize=True,
    )
    ml_10m = pd.read_parquet("./ml-10M100K/ratings.parquet")
    num_users = ml_10m["user_id"].unique() + 1
    num_items = ml_10m["item_id"].unique() + 1
    data = NCFData(ml_10m, num_negatives=4, num_negatives_test=1, batch_size=1028)
