import random
import warnings

from tqdm import tqdm
from loguru import logger

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")


class UserItemRatingDataset(Dataset):
    def __init__(self, user: list, item: list, rating: list):
        super(UserItemRatingDataset, self).__init__()

        self.user = torch.tensor(user, dtype=torch.long)
        self.item = torch.tensor(item, dtype=torch.long)
        self.target = torch.tensor(rating, dtype=torch.float)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.user[idx], self.item[idx], self.target[idx]


class NCFData(object):
    def __init__(self, ratings, num_negatives, num_negatives_test, batch_size: int):
        logger.info("Start init SCFData")
        self.ratings = ratings
        self.num_negatives = num_negatives
        self.num_negatives_test = num_negatives_test
        self.batch_size = batch_size

        self.preprocess_ratings = self._reindex(self.ratings)
        self.user_pool = set(self.ratings["user_id"].unique())
        self.item_pool = set(self.ratings["item_id"].unique())

        self.train_ratings, self.test_ratings = self._leave_p_out(
            self.preprocess_ratings
        )
        self.negatives = self._negative_sampling(self.preprocess_ratings)

    def _reindex(self, ratings):
        logger.info("Reindex data")
        user = list(ratings["user_id"].drop_duplicates())
        self.user2id = {w: i for i, w in enumerate(user)}

        item = list(ratings["item_id"].drop_duplicates())
        self.item2id = {w: i for i, w in enumerate(item)}

        ratings["user_id"] = ratings["user_id"].apply(lambda x: self.user2id[x])
        ratings["item_id"] = ratings["item_id"].apply(lambda x: self.item2id[x])
        ratings["rating"] = ratings["rating"].apply(lambda x: float(x > 0))

        return ratings

    def _leave_one_out(self, ratings):
        ratings["rank_latest"] = ratings.groupby(["user_id"])["timestamp"].rank(
            method="first", ascending=False
        )
        test = ratings.loc[ratings["rank_latest"] == 1]
        train = ratings.loc[ratings["rank_latest"] > 1]

        return (
            train[["user_id", "item_id", "rating"]],
            test[["user_id", "item_id", "rating"]],
        )

    def _leave_p_out(self, ratings, p=10):
        logger.info("Start leave P out")
        ratings["rank_latest"] = ratings.groupby(["user_id"])["timestamp"].rank(
            method="first", ascending=False
        )
        test = ratings.loc[ratings["rank_latest"] < p + 1]
        train = ratings.loc[ratings["rank_latest"] > p]
        return (
            train[["user_id", "item_id", "rating"]],
            test[["user_id", "item_id", "rating"]],
        )

    def _negative_sampling(self, ratings):
        logger.info("Start negative sampling")
        interact_status = (
            ratings.groupby("user_id")["item_id"]
            .apply(set)
            .reset_index()
            .rename(columns={"item_id": "interacted_items"})
        )

        interact_status["negative_samples"] = [
            random.sample(
                self.item_pool - interact_status["interacted_items"].iloc[i],
                self.num_negatives_test,
            )
            for i in range(interact_status["interacted_items"].shape[0])
        ]

        return interact_status[["user_id", "negative_samples", "interacted_items"]]

    def get_train_instance(self):
        users, items, ratings = [], [], []
        train_ratings_negatives = pd.DataFrame()
        train_ratings_negatives["user_id"] = self.negatives["user_id"]
        train_ratings_negatives["negatives"] = [
            random.sample(
                self.item_pool - self.negatives["interacted_items"].iloc[i],
                len(self.negatives["interacted_items"].iloc[i])
                if len(self.item_pool)
                - 2 * len(self.negatives["interacted_items"].iloc[i])
                > 0
                else self.num_negatives,
            )
            for i in range(self.negatives["interacted_items"].shape[0])
        ]
        train_ratings_negatives = train_ratings_negatives.explode("negatives")

        users = np.append(
            self.train_ratings["user_id"], train_ratings_negatives["user_id"]
        ).astype(np.int64)
        items = np.append(
            self.train_ratings["item_id"], train_ratings_negatives["negatives"]
        ).astype(np.int64)
        ratings = np.append(
            self.train_ratings["rating"],
            [0 for i in range(train_ratings_negatives.shape[0])],
        ).astype(np.int64)

        assert len(users) == len(items) and len(items) == len(ratings)

        dataset = UserItemRatingDataset(user=users, item=items, rating=ratings)
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def get_test_instance(self, p=10):
        users, items, ratings = [], [], []
        test_ratings = pd.merge(
            self.test_ratings,
            self.negatives[["user_id", "negative_samples"]],
            on="user_id",
        )

        for user in tqdm(np.unique(test_ratings["user_id"])):
            for row in (test_ratings.loc[test_ratings["user_id"] == user]).itertuples():
                users.append(int(row.user_id))
                items.append(int(row.item_id))
                ratings.append(float(row.rating))
            for item_negative in test_ratings.loc[test_ratings["user_id"] == user][
                "negative_samples"
            ].iloc[0]:
                users.append(int(user))
                items.append(int(item_negative))
                ratings.append(float(0))

        dataset = UserItemRatingDataset(user=users, item=items, rating=ratings)
        return DataLoader(
            dataset,
            batch_size=self.num_negatives_test + p,
            shuffle=False,
            num_workers=4,
        )
