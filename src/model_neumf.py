import torch

import torch.nn as nn


class NeuMF(nn.Module):
    def __init__(self,
                 num_users,
                 num_items,
                 embedding_dim,
                 layers,
                 layers_neumf):
        super(NeuMF, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.layers = layers
        self.layers_neumf = layers_neumf

        self.embedding_user_mlp = nn.Embedding(
            num_embeddings=self.num_users + 1,
            embedding_dim=self.embedding_dim,
            padding_idx=0
        )
        self.embedding_item_mlp = nn.Embedding(
            num_embeddings=self.num_items + 1,
            embedding_dim=self.embedding_dim,
            padding_idx=0
        )

        self.embedding_user_mf = nn.Embedding(
            num_embeddings=self.num_users + 1,
            embedding_dim=self.embedding_dim,
            padding_idx=0
        )
        self.embedding_item_mf = nn.Embedding(
            num_embeddings=self.num_items + 1,
            embedding_dim=self.embedding_dim,
            padding_idx=0
        )

        self.fc = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.layers[0]),
            nn.BatchNorm1d(self.layers[0]),
            nn.GELU(),
            nn.Linear(self.layers[0], self.layers[1]),
            nn.BatchNorm1d(self.layers[1]),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(self.layers[1], self.layers[2]),
            nn.BatchNorm1d(self.layers[2]),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(self.layers[2], self.layers[3]),
            nn.BatchNorm1d(self.layers[3]),
            nn.GELU(),
        )
        
        # self.fc_neumf = nn.Sequential(
        #     nn.Linear(self.layers[-1] + self.embedding_dim, self.layers_neumf[0]),
        #     nn.BatchNorm1d(self.layers_neumf[0]),
        #     nn.ReLU(),
        #     nn.Linear(self.layers_neumf[0], self.layers_neumf[1]),
        #     nn.BatchNorm1d(self.layers_neumf[1]),
        #     nn.ReLU(),
        #     nn.Linear(self.layers_neumf[1], self.layers_neumf[2]),
        #     nn.BatchNorm1d(self.layers_neumf[2]),
        #     nn.ReLU()
        # )
        
        self.affine_output = nn.Linear(
            self.layers[-1] + self.embedding_dim, 1
        )
        
        self.activate = nn.Sigmoid()
        
        nn.init.xavier_uniform_(self.embedding_user_mlp.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.embedding_item_mlp.weight, gain=nn.init.calculate_gain('relu'))

        nn.init.xavier_uniform_(self.embedding_user_mf.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.embedding_item_mf.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, user_indices, item_indices):
        # Эмбеддинги для mlp
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        # Эмбеддинги для mf
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)
        
        
        element_product_mf = torch.mul(
            user_embedding_mf, item_embedding_mf
        )
        element_product_mlp = torch.cat(
            (user_embedding_mlp, item_embedding_mlp), -1
        )

        layers_mlp = self.fc(element_product_mlp)
        
        logits = self.affine_output(torch.cat(
                (layers_mlp, element_product_mf), -1))
        rating = self.activate(logits)

        return rating
        