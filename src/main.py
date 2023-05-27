import sys
import pandas as pd
import torch
import torch.nn as nn
import argparse

from tqdm import trange
from loguru import logger
from numpy import mean

from preprocessing import NCFData
from metrics import metrics
from model_neumf import NeuMF
from utils import data2excel

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_pipeline(model, optimizer, criterion, data, num_epoch):
    loss_history_epoch = []
    metrics_history = {
        "Test_loss": [],
        "HR@10": [],
        "Precision@10": [],
        "Recall@10": [],
        "MRR@10": [],
        "MAP@10": [],
        "NDCG@10": [],
    }
    test_loader = data.get_test_instance()

    for epoch in trange(num_epoch):
        loss_history = []
        model.train()

        train_loader = data.get_train_instance()

        for user, item, label in train_loader:
            user = user.to(DEVICE)
            item = item.to(DEVICE)
            label = label.to(DEVICE)

            optimizer.zero_grad()
            prediction = model(user, item)

            loss = criterion(
                prediction.view(-1).to(torch.float64), label.to(torch.float64)
            )
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

        model.eval()
        test_loss, hr_i, precision_i, recall_i, mrr_i, map_i, ndcg_i = metrics(
            model, test_loader, criterion, 10, DEVICE
        )
        metrics_history["Test_loss"].append(test_loss)
        metrics_history["HR@10"].append(hr_i)
        metrics_history["Precision@10"].append(precision_i)
        metrics_history["Recall@10"].append(recall_i)
        metrics_history["MRR@10"].append(mrr_i)
        metrics_history["MAP@10"].append(map_i)
        metrics_history["NDCG@10"].append(ndcg_i)
        loss_history_epoch.append(mean(loss_history))

        print(
            f"[Epoch {epoch}]| Loss train: {loss_history_epoch[-1]:.5f}\tLoss test: {test_loss}\n"
            f"HR@10: {hr_i:.3f}\tPrecision@10: {precision_i:.3f}\tRecall@10: {recall_i:.3f}\t"
            f"MRR@10: {mrr_i:.3f}\tMAP@10: {map_i:.3f}\tNDCG@10 {ndcg_i:.3f} |"
        )

    return loss_history_epoch, metrics_history


def main():
    logger.info("Reading file")
    ml_10m = pd.read_parquet("./ratings.parquet")
    num_users = ml_10m["user_id"].nunique() + 1
    num_items = ml_10m["item_id"].nunique() + 1
    data = NCFData(ml_10m, num_negatives=4, num_negatives_test=100, batch_size=1024)
    logger.info(f"Model to {DEVICE}")
    model = NeuMF(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=64,
        layers=[1024, 512, 256, 128],
        layers_neumf=[512, 256, 128],
    )
    model.to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    logger.info("Start learning model")
    loss_history_neumf, metrics_history_neumf = train_pipeline(
        model, optimizer, criterion, data, 20
    )

    data2excel(
        "neumf_25m_1024_4neg_100negtest_notwolayers",
        loss_history_neumf,
        metrics_history_neumf,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("input_file", help="Путь к входному файлу parquet")
    parser.add_argument(
        "-o", "--output_file", help="Путь к выходному файлу с лоссами и метриками"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Количество эпох для обучения"
    )
    parser.add_argument(
        "--layers", type=list, default=[1024, 512, 256, 128], help="Слои части MLP"
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(
        sys.__stdout__,
        format="[{time:YYYY-MM-DD HH:mm:ss}] {level} | {message}",
        level="TRACE",
        colorize=True,
    )

    main()
