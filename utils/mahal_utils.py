# mahal_utils.py
'''
Helper functions for mahalanobis calculations.

mahalanobis_distance = sqrt((x-mu).T*sigma^-1*(x-mu)

'''

import numpy as np
import torch
from scipy.spatial.distance import mahalanobis


class Mahalanobis(object):
    def __init__(self, calibration_embeddings, n_class, n_cal):
        self.calibration_embeddings = calibration_embeddings.view(n_class, n_cal, calibration_embeddings.shape[-1])
        self.n_class, self.n_calibrate, self.z_dim = n_class, n_cal, calibration_embeddings.size()[-1]
        self.mu_0 = self._get_whole_mean()
        self.sigma_0 = self._get_whole_sigma()

    def mahalanobis_distance(self, z, mu, sigma, diagonal=False):
        # embs = NXD
        # mu = MXD
        # SIGMA = DXD
        n = z.size(0) * z.size(1)
        d = z.size(2)
        m = mu.size(0)
        z = z.squeeze().view(n, d)
        mu_k = mu.squeeze().view(m, d)
        z = z.unsqueeze(1).expand(n, m, d)
        mu_k = mu_k.unsqueeze(0).expand(n, m, d)
        # Then (m classes of : class*examples of dim D) X (DXD) * (m classes of: dim D of class*examples) = (m classes of class*examples).diag() = m distances for each n_examples
        if diagonal:
            # fuckin multipy together! **debugger+tries every dimension combination*** =>
            # **transposes 0 & 1** => ***multiplies *** => :0

            radicand = torch.matmul(torch.matmul((z - mu_k).transpose(0, 1), torch.linalg.pinv(sigma)),
                                    (z - mu_k).transpose(0, 1).mT)#.diagonal(dim1=1, dim2=2).T
            return radicand.view(n,n,m).diagonal()

            radicand = torch.matmul(torch.matmul((z - mu_k).transpose(0, 1), torch.linalg.pinv(sigma)),
                                    (z - mu_k).transpose(0, 1).mT).diagonal(dim1=1, dim2=2).T
            return torch.sqrt(radicand)
        else:
            return torch.matmul(torch.matmul((z - mu_k), torch.linalg.pinv(sigma)),
                                (z - mu_k).mT).diagonal(dim1=1, dim2=2)

    def _calc_mahalanobis_distance(self):
        return self.mahalanobis_distance(self.calibration_embeddings, self._get_class_means(), self._get_class_sigma())

    def _calc_overall_distance(self):
        return self.mahalanobis_distance(self.calibration_embeddings, self.mu_0, self.sigma_0)

    def relative_mahalanobis_distance(self, z):
        temp = self.calibration_embeddings
        self.calibration_embeddings = z
        md_k = self._calc_mahalanobis_distance()
        md_0 = self._calc_overall_distance()
        self.calibration_embeddings = temp
        return md_k - md_0

    def diag_mahalanobis_distance(self, z_test):

        z = z_test.view(z_test.size(0) * z_test.size(1), -1)
        features = z.view(z.size(0), z.size(1), -1)
        features = torch.mean(features, 2).unsqueeze(0)
        md_k = []

        for clazz in range(self.n_class):
            score = self.mahalanobis_distance(features, self._get_class_means()[clazz], self._get_class_sigma(), True)
            md_k.append(score.view(-1, 1))

        return torch.cat(md_k, 1)

        diag_sigma_k = torch.diag_embed(self._get_class_sigma_k().diagonal(dim1=-2, dim2=-1))
        dk = self.mahalanobis_distance(z_test, self._get_class_means(), diag_sigma_k, diagonal=True)
        return dk

    def _get_class_means(self) -> torch.Tensor:
        """
        :return: mean embedding for each support class
        """
        mu_k = self.calibration_embeddings.mean(1, keepdim=True)
        return mu_k

    def _get_whole_mean(self) -> torch.Tensor:
        """
        :return: Mean embedding of entire dataset
        """
        mu_naught = self.calibration_embeddings.mean(1, keepdim=True).mean(0, keepdim=True)
        return mu_naught

    def _get_class_sigma_k(self):
        centred = self.calibration_embeddings - self._get_class_means()
        sigma_k = torch.matmul(centred.transpose(-2, -1), centred)
        return sigma_k

    def _get_class_sigma(self):
        sigma = self._get_class_sigma_k().sum(dim=0, keepdim=True)
        return sigma

    def _get_whole_sigma(self):
        centred = self.calibration_embeddings - self._get_whole_mean()
        sigma_0 = torch.matmul(centred.transpose(-2, -1), centred).sum(dim=0, keepdim=True)
        return sigma_0


class RelativeMahalanobis(object):
    def __init__(self, embeddings):
        self.z = embeddings
        self.n_class = self.z.size()[0]
        self.n_support = self.z.size()[1]
        self.z_dim = self.z.size()[2]
        self.mu_k = torch.div(torch.sum(self.z, dim=1, keepdim=True), self.n_support)
        self.mu_0 = torch.mean(torch.mean(self.z, dim=0, keepdim=True), dim=1, keepdim=True)
        self.centred_class_z = self.z - self.mu_k
        self.centred_z = self.z - self.mu_0
        self.sigma = torch.div(torch.sum(
            torch.matmul(self.centred_class_z.transpose(1, 2), self.centred_class_z), dim=0),
            self.n_class * self.n_support)
        self.sigma_0 = torch.div(torch.sum(
            torch.matmul(self.centred_z.transpose(1, 2), self.centred_z), dim=0),
            self.n_class * self.n_support)  # [z_dim, z_dim]
        self.sigma_k = []
        for i in range(self.n_class):
            self.sigma_k.append(
                torch.div(
                    torch.mm((self.z[i] - self.mu_k[i]).T, (self.z[i] - self.mu_k[i])),
                    self.n_support)
            )
        self.sigma_k = torch.cat(self.sigma_k, dim=0).view(self.n_class, self.z_dim, self.z_dim)

    def rel_mahalanobis(self, z):
        centred_class_z = z - self.mu_k
        centred_z = z - self.mu_0
        radicand = torch.matmul(torch.matmul(centred_class_z, torch.linalg.pinv(self.sigma)),
                                centred_class_z.transpose(1, 2))
        radicand_0 = torch.matmul(torch.matmul(centred_z, torch.linalg.pinv(self.sigma_0)),
                                  centred_z.transpose(1, 2))
        mahal = torch.sqrt(torch.clamp(radicand, min=1e-12))
        mahal_0 = torch.sqrt(torch.clamp(radicand_0, min=1e-12))

        return torch.diagonal(mahal - mahal_0).T

    def _mahalanobis(self, z, k):
        centred_class_z = z - self.mu_k[k]
        diag_sigma = torch.diag_embed(torch.diagonal(self.sigma_k[k], dim1=-2, dim2=-1))
        radicand = torch.matmul(torch.matmul(centred_class_z, torch.linalg.pinv(diag_sigma)),
                                centred_class_z.T)
        score = torch.sqrt(torch.clamp(radicand, min=1e-12))
        return torch.diagonal(score)  # .transpose(-1, -2)

    def mahalanobis(self, z):
        md_k = []
        for i in range(self.n_class):
            dist = self._mahalanobis(z, i)
            md_k.append(dist)
        return torch.cat(md_k, dim=0).view(self.n_class, int(z.size()[0] / self.n_class), -1)


"""
OLD RMD: WORKS THE SAME AS NEW RMD (IE DOESNT WORK)
"""


class RMD(object):
    def __init__(self, z, n_class, z_dim, n_support):
        self.z = z
        self.n_class = n_class
        self.z_dim = z_dim
        self.n_support = n_support
        self.mu_0, self.mu_k = self._get_mean_embeddings()
        self.sigma, self.sigma_0, self.sigma_k = self._get_sigmas()

    def batch_relative_mahalanobis_distance(self, z):
        # z shape: n_class, n_cal, z_dim
        # mu_k shape: n_class, z_dim
        # Calc x-mu
        n_class = self.mu_k.shape[0]

        z_dim = self.z_dim
        z = z.view(n_class, -1, z_dim)
        n_cal = z.shape[-2]
        intermediate_k = (z.transpose(0, 1) - self.mu_k).transpose(0, 1).unsqueeze(-1)
        intermediate_0 = (z.transpose(0, 1) - self.mu_0).transpose(0, 1).unsqueeze(-1)
        # Shape [64,64]
        test_against = mahalanobis(z[0][0], self.mu_k[0], torch.linalg.pinv(self.sigma))
        test_against = np.nan_to_num(test_against, nan=0.0)

        m_k = torch.sqrt(torch.clamp(
            torch.matmul(torch.matmul(intermediate_k.transpose(-1, -2), torch.linalg.pinv(self.sigma)), intermediate_k),
            min=1e-12))
        m_0 = torch.sqrt(torch.clamp(
            torch.matmul(torch.matmul(intermediate_0.transpose(-1, -2), torch.linalg.pinv(self.sigma_0)),
                         intermediate_0), min=1e-12))
        m_k_rel = m_k - m_0
        assert torch.allclose(torch.Tensor([test_against]), m_k[0][0], atol=5e-1), f"{test_against} != {m_k[0][0]}"
        assert m_k_rel.shape == torch.Size([n_class, n_cal, 1, 1]), f"Size is {m_k_rel.shape}"
        return m_k_rel

    def diag_mahalanobis_dist(self, z_test):
        # z shape:  n_query, z_dim
        # mu_k shape: n_class, z_dim

        z_dim = z_test.size()[-1]
        n_class = self.mu_k.shape[0]
        n_query = z_test.size()[-2]
        z = z_test.unsqueeze(0)
        intermediate_k = (z_test.transpose(0, 1) - self.mu_k).transpose(0, 1).unsqueeze(-1)
        # TODO: Try rel dist here
        # intermediate_k has shape: [n_class, n_query, z_dim, 1]
        diag_sigma = torch.diag_embed(torch.diagonal(self.sigma_k, dim1=-2, dim2=-1))
        # Diag sigma has shape: [5,64,64]
        pinv = torch.linalg.pinv(diag_sigma)
        test_against = mahalanobis(z.squeeze()[0][0], self.mu_k[0], pinv[0])
        pinv = pinv.unsqueeze(1)
        # pinv has shape [5,1,64,64]
        left = torch.matmul(intermediate_k.transpose(-1, -2), pinv)
        m_k = torch.sqrt(torch.clamp(torch.matmul(left, intermediate_k), min=1e-12)).squeeze()
        assert m_k.size() == torch.Size([n_class, n_query]), f"Size is {m_k.size()}"
        assert torch.allclose(m_k[0][0], torch.Tensor([test_against]), atol=0.5), f"{m_k[0][0]} != {test_against}"
        return m_k

    def _get_mean_embeddings(self):
        """
        u_k = 1/n_class * sum_over_supports of embedding of class k.
        """
        # obtain mu_k and mu_0:
        # z_suppoprt shape: [n_class, n_support, z_dim]
        assert [self.z.size()[0], self.z.size()[2]] == [self.n_class, self.z_dim], f"{self.z.size()}"
        mu_k = self.z.mean(1)
        assert mu_k.size() == torch.Size(
            [self.n_class, self.z_dim]), f"Size is {mu_k.size()}, expecting size {[self.n_class, self.z_dim]}"
        mu_0 = mu_k.mean(0)
        assert mu_0.size() == torch.Size([self.z_dim])
        #      assert torch.allclose(mu_0[0], self.z.mean(0).mean(0)[0], atol=0.00001), f"{mu_0[0]} != {self.z.mean(0).mean(0)[0]}"
        return mu_0, mu_k

    def _get_sigmas(self):
        # Obtain Covariance Matrices sigma_k and sigma:
        intermediate = (self.z.transpose(0, 1) - self.mu_k).transpose(0, 1).unsqueeze(
            3)  # Prepare to multiply the modified vectors of size [z_dim,1]
        sigma_c_k = torch.matmul(intermediate, intermediate.transpose(-1, -2))
        assert sigma_c_k.size() == torch.Size([self.n_class, self.n_support, self.z_dim, self.z_dim])
        sigma_k = sigma_c_k.mean(1)
        assert sigma_k.size() == torch.Size([self.n_class, self.z_dim, self.z_dim])
        torch_sigma_k = torch.cov(self.z[0].T)
        #        assert torch.allclose(torch_sigma_k, sigma_k[0], atol=0.1), f"{sigma_k[0]} != {torch_sigma_k}"
        sigma = sigma_k.mean(0)
        assert sigma.size() == torch.Size([self.z_dim, self.z_dim])
        # Obtain Cov Matrices sigma_0
        intermediate = ((self.z.transpose(0, 1) - self.mu_0).transpose(0, 1).unsqueeze(3))
        sigma_0_c_k = torch.matmul(intermediate, intermediate.transpose(-1, -2))
        sigma_0 = sigma_0_c_k.mean(1).mean(0)
        assert sigma_0.size() == torch.Size([self.z_dim, self.z_dim])
        return sigma, sigma_0, sigma_k






