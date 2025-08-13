from abc import abstractmethod
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from easyfsl.methods.utils import compute_prototypes
from utils.mahal_utils import Mahalanobis
from scipy.stats import gaussian_kde
from scipy.integrate import quad
import numpy as np
from torchmetrics.classification import MulticlassCalibrationError, MulticlassF1Score, MulticlassConfusionMatrix
class FewShotClassifier(nn.Module):
    """
    Abstract class providing methods usable by all few-shot classification algorithms
    """

    def __init__(
        self,
        backbone: Optional[nn.Module] = None,
        use_softmax: bool = False,
        feature_centering: Optional[Tensor] = None,
        feature_normalization: Optional[float] = None,
    ):
        """
        Initialize the Few-Shot Classifier
        Args:
            backbone: the feature extractor used by the method. Must output a tensor of the
                appropriate shape (depending on the method).
                If None is passed, the backbone will be initialized as nn.Identity().
            use_softmax: whether to return predictions as soft probabilities
            feature_centering: a features vector on which to center all computed features.
                If None is passed, no centering is performed.
            feature_normalization: a value by which to normalize all computed features after centering.
                It is used as the p argument in torch.nn.functional.normalize().
                If None is passed, no normalization is performed.
        """
        super().__init__()

        self.backbone = backbone if backbone is not None else nn.Identity()
        self.use_softmax = use_softmax

        self.prototypes = torch.tensor(())
        self.support_features = torch.tensor(())
        self.support_labels = torch.tensor(())
        self.rmd = None

        self.feature_centering = (
            feature_centering if feature_centering is not None else torch.tensor(0)
        )
        self.feature_normalization = feature_normalization

    @abstractmethod
    def forward(
        self,
        query_images: Tensor,
    ) -> Tensor:
        """
        Predict classification labels.
        Args:
            query_images: images of the query set of shape (n_query, **image_shape)
        Returns:
            a prediction of classification scores for query images of shape (n_query, n_classes)
        """
        raise NotImplementedError(
            "All few-shot algorithms must implement a forward method."
        )

    def process_support_set(
        self,
        support_images: Tensor,
        support_labels: Tensor,
    ):
        """
        Harness information from the support set, so that query labels can later be predicted using a forward call.
        The default behaviour shared by most few-shot classifiers is to compute prototypes and store the support set.
        Args:
            support_images: images of the support set of shape (n_support, **image_shape)
            support_labels: labels of support set images of shape (n_support, )
        """
        self.compute_prototypes_and_store_support_set(support_images, support_labels)

    @staticmethod
    def is_transductive() -> bool:
        raise NotImplementedError(
            "All few-shot algorithms must implement a is_transductive method."
        )

    def compute_features(self, images: Tensor) -> Tensor:
        """
        Compute features from images and perform centering and normalization.
        Args:
            images: images of shape (n_images, **image_shape)
        Returns:
            features of shape (n_images, feature_dimension)
        """
        original_features = self.backbone(images)
        centered_features = original_features - self.feature_centering
        if self.feature_normalization is not None:
            return nn.functional.normalize(
                centered_features, p=self.feature_normalization, dim=1
            )
        return centered_features

    def softmax_if_specified(self, output: Tensor, temperature: float = 1.0) -> Tensor:
        """
        If the option is chosen when the classifier is initialized, we perform a softmax on the
        output in order to return soft probabilities.
        Args:
            output: output of the forward method of shape (n_query, n_classes)
            temperature: temperature of the softmax
        Returns:
            output as it was, or output as soft probabilities, of shape (n_query, n_classes)
        """
        return (temperature * output).softmax(-1) if self.use_softmax else output

    def l2_distance_to_prototypes(self, samples: Tensor) -> Tensor:
        """
        Compute prediction logits from their euclidean distance to support set prototypes.
        Args:
            samples: features of the items to classify of shape (n_samples, feature_dimension)
        Returns:
            prediction logits of shape (n_samples, n_classes)
        """
        return -torch.cdist(samples, self.prototypes)

    def cosine_distance_to_prototypes(self, samples) -> Tensor:
        """
        Compute prediction logits from their cosine distance to support set prototypes.
        Args:
            samples: features of the items to classify of shape (n_samples, feature_dimension)
        Returns:
            prediction logits of shape (n_samples, n_classes)
        """
        return (
            nn.functional.normalize(samples, dim=1)
            @ nn.functional.normalize(self.prototypes, dim=1).T
        )

    def compute_prototypes_and_store_support_set(
        self,
        support_images: Tensor,
        support_labels: Tensor,
    ):
        """
        Extract support features, compute prototypes, and store support labels, features, and prototypes.
        Args:
            support_images: images of the support set of shape (n_support, **image_shape)
            support_labels: labels of support set images of shape (n_support, )
        """
        self.support_labels = support_labels
        self.support_features = self.compute_features(support_images)
        self._raise_error_if_features_are_multi_dimensional(self.support_features)
        self.prototypes = compute_prototypes(self.support_features, support_labels)

    def compute_kernels_from_calibration_set(self, cal_data: Tensor, cal_labels: Tensor, sup_data: Tensor, sup_labels: Tensor, n_class, n_cal, bandwidth):
        #TODO: April 25, I change to using support data to compute mahalanbois, rather than query data.
        train_prototypes= self.prototypes
        z_dim = len(self.backbone.state_dict()["2.weight"][-1])
        #TODO: May have to make target_inds as I did in protonet
        target_inds = cal_labels
        cal_features = self.compute_features(cal_data)
        self.compute_prototypes_and_store_support_set(sup_data, sup_labels)
        n_sup = int(self.support_features.shape[0] / n_class)
        input_to_mahalanobis = self.prototypes.expand(n_class, n_class, z_dim) # FROM ALL PREVIOUS EXPERIMENTS
        #input_to_mahalanobis = self.compute_features(sup_data)
        self.rmd = Mahalanobis(input_to_mahalanobis, n_class, n_sup)  # Now compute mahalanobis # USE support to set mahal vars
        # use calibrate to compute rel_mahal (if using sup, ood scores at test time will be higher
        m_k_rel = (torch.min(self.rmd.relative_mahalanobis_distance(cal_features.view(n_class, n_cal, -1)), dim=1).
                   values.view(n_class, n_cal))
        self.prototypes = train_prototypes
        m_k_rel = m_k_rel.detach().numpy()

        # Obtain n_class Gaussian KDE's for each class
        def get_kde(rel_mahalanobis, target_inds, n_way):
            class_kernel_densities = [0 for _ in range(n_way)]
            for idx in range(len(class_kernel_densities)):
                # TODO: Sometimes gives singular covariance matrix error, must handle
                class_kernel_densities[idx] = gaussian_kde(rel_mahalanobis[idx], bw_method=bandwidth)
            return class_kernel_densities
        g_k = get_kde(m_k_rel, target_inds, n_class)
        return g_k  # Used in alg 3!

    def ood_test_alg(self, samples: Tensor, sample_labels: Tensor, n_class, n_query, g_k):
        target_inds = sample_labels
        z = self.compute_features(samples)
        z = z.view(n_class, n_query, -1)
        z_dim = z.shape[-1]
        dists = self.rmd.diag_mahalanobis_distance(z)

        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)
        _, y_hat = log_p_y.max(2)

        rel_d = torch.min(self.rmd.relative_mahalanobis_distance(z), dim=1).values.view(n_class, n_query)
        pvals = []
        mid_pvals = []
        correct_preds = 0
        # OOD Scoring
        for i in range(n_class):
            for index in range(len(y_hat.tolist()[i])):
                predicted_class = y_hat.tolist()[i][index]
                if i == predicted_class: correct_preds += 1
                r = rel_d.tolist()[predicted_class][index]
                max_val = g_k[predicted_class].dataset.max()
                bw = g_k[predicted_class].factor
                grid = np.linspace(r, max_val + bw, 10000)
                p_val = np.trapz(g_k[predicted_class].pdf(grid), grid, dx=bw / 2)
                mid_pvals.append(1 - p_val)  # confidence score is 1 minus the p value
            pvals.append(mid_pvals)

        # Calc stats

        calibration_error = MulticlassCalibrationError(num_classes=n_class, n_bins=n_class, norm='l1')
        micro_f1 = MulticlassF1Score(num_classes=n_class, average='micro')
        caliber = calibration_error(log_p_y.view(n_class * n_query, -1), target_inds.flatten())
        micro_f = micro_f1(log_p_y.view(n_class * n_query, -1), target_inds.flatten())
        confusion_matrix = MulticlassConfusionMatrix(num_classes=n_class)
        confusion = confusion_matrix(log_p_y.view(n_class * n_query, -1), target_inds.flatten())
        # print("Expected Calibration Error is: ", caliber)
        # print("Micro F1 Score is: ", micro_f)
        acc_vals = torch.eq(y_hat.view(target_inds.shape[0]), target_inds).float().mean()
        # print("Accuracy:", correct_preds/(n_class*n_query))
        return mid_pvals, acc_vals, caliber, micro_f, confusion

    def ood_score_sample(self, sample, g_k):
        n_examples = sample.size(0)
        z = self.compute_features(sample)
        z = z.unsqueeze(0)
        dists = self.rmd.diag_mahalanobis_distance(z)
        log_p_y = F.log_softmax(-dists, dim=1).view(n_examples, -1)
        _, y_hat = log_p_y.max(1)
        rel_d = torch.min(self.rmd.relative_mahalanobis_distance(z), dim=1).values.view(-1, n_examples)
        pvals = []
        mid_pvals = []
        correct_preds = 0
        # OOD Scoring

        for index in range(len(y_hat.tolist())):
            predicted_class = y_hat.tolist()[index]
            r = rel_d.tolist()[0][index]
            max_val = g_k[predicted_class].dataset.max()
            bw = g_k[predicted_class].factor
            grid = np.linspace(r, max_val+bw, 100000)
            p_val = np.trapz(g_k[predicted_class].pdf(grid), grid, dx=bw/2)
            mid_pvals.append(1 - p_val)  # confidence score is 1 minus the p value
        pvals.append(mid_pvals)
        return pvals



    @staticmethod
    def _raise_error_if_features_are_multi_dimensional(features: Tensor):
        if len(features.shape) != 2:
            raise ValueError(
                "Illegal backbone or feature shape. "
                "Expected output for an image is a 1-dim tensor."
            )
