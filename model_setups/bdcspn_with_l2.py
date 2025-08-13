from easyfsl.methods import BDCSPN
from torch import Tensor, nn
import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm

class BDLPN(BDCSPN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def rectify_prototypes(self, query_features: Tensor):
        """
        Updates prototypes with label propagation and feature shifting.
        Args:
            query_features: query features of shape (n_query, feature_dimension)
        """
        n_classes = self.support_labels.unique().size(0)
        one_hot_support_labels = nn.functional.one_hot(self.support_labels, n_classes)

        average_support_query_shift = self.support_features.mean(
            0, keepdim=True
        ) - query_features.mean(0, keepdim=True)
        query_features = query_features + average_support_query_shift

        support_logits = self.l2_distance_to_prototypes(self.support_features).exp()
        query_logits = self.l2_distance_to_prototypes(query_features).exp()

        one_hot_query_prediction = nn.functional.one_hot(
            query_logits.argmax(-1), n_classes
        )

        normalization_vector = (
            (one_hot_support_labels * support_logits).sum(0)
            + (one_hot_query_prediction * query_logits).sum(0)
        ).unsqueeze(
            0
        )  # [1, n_classes]
        support_reweighting = (
            one_hot_support_labels * support_logits
        ) / normalization_vector  # [n_support, n_classes]
        query_reweighting = (
            one_hot_query_prediction * query_logits
        ) / normalization_vector  # [n_query, n_classes]

        self.prototypes = (support_reweighting * one_hot_support_labels).t().matmul(
            self.support_features
        ) + (query_reweighting * one_hot_query_prediction).t().matmul(query_features)

    def proto_rectification(self, support, query, shot):
        """
            Other p rect imp from : https://github.com/oveilleux/Realistic_Transductive_Few_Shot/blob/master/src/methods/bdcspn.py
            inputs:
                support : np.Array of shape [n_task, s_shot, feature_dim]
                query : np.Array of shape [n_task, q_shot, feature_dim]
                shot: Shot

            ouput:
                proto_weights: prototype of each class
        """
        eta = support.mean(1) - query.mean(1)  # Shifting term
        query = query + eta[:, np.newaxis, :]  # Adding shifting term to each normalized query feature
        query_aug = torch.concatenate((support, query), axis=1)  # Augmented set S' (X')
        support_ = support.reshape(support.shape[0], 1, len(self.support_labels.unique()), support.shape[-1]).mean(
            1)  # Init basic prototypes Pn
        #support_ = torch.from_numpy(support_)
        #query_aug = torch.from_numpy(query_aug)

        proto_weights = []
        for j in range(len(self.support_labels.unique())):
            distance = self.cosine_distance_to_prototypes(query_aug[j])
            predict = torch.argmin(distance, dim=1)
            l2_sim = F.cosine_similarity(query_aug[j][:, None, :], support_[j][None, :, :], dim=2)  # Cosine similarity between X' and Pn
            #cos_sim = 10 * cos_sim
            W = F.softmax(10*l2_sim, dim=1)
            support_list = [(W[predict == i, i].unsqueeze(1) * query_aug[j][predict == i]).mean(0, keepdim=True) for i
                            in predict.unique()]
            proto = torch.cat(support_list, dim=0)  # Rectified prototypes P'n
            proto_weights.append(proto.mean(0))
        proto_weights = torch.stack(proto_weights, dim=0)
        return proto_weights

    def forward(
            self,
            query_images: Tensor,
    ) -> Tensor:
        """
        Overrides forward method of FewShotClassifier.
        Update prototypes using query images, then classify query images based
        on their cosine distance to updated prototypes.
        """
        num_classes = self.support_labels.unique().size(0)
        num_queries = query_images.size(0) // num_classes
        num_sups = self.support_features.size(0) // num_classes
        query_features = self.compute_features(query_images)
        # TODO: AM I sure they rectifying the prototypes correctly??
        self.rectify_prototypes(
            query_features=query_features,
        )


        return self.softmax_if_specified(
            self.l2_distance_to_prototypes(query_features)
        )
