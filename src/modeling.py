"""双数值特征的加性模型：不包含剂量与温度的交互项。"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.config import RANDOM_SEED


class AdditiveRBF(RBF):
    """按维度分别计算 RBF 后求和，而不是相乘，避免隐含交互。"""

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        if eval_gradient and Y is not None:
            raise ValueError("仅自协方差支持梯度计算。")
        Y = X if Y is None else np.atleast_2d(Y)
        scales = np.broadcast_to(self.length_scale, (X.shape[1],))
        squared = ((X[:, None, :] - Y[None, :, :]) / scales) ** 2
        components = np.exp(-0.5 * squared)
        covariance = components.sum(axis=2)
        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                gradient = np.empty((*covariance.shape, 0))
            else:
                gradient = components * squared
                if not self.anisotropic:
                    gradient = gradient.sum(axis=2, keepdims=True)
            return covariance, gradient
        return covariance

    def diag(self, X):
        """加性核的对角线等于维度数。"""
        return np.full(len(X), np.asarray(X).shape[1], dtype=float)


def additive_quadratic(X):
    """仅添加剂量平方作为对照，温度保持线性，不构造交互项。"""
    values = np.asarray(X, dtype=float)
    return np.column_stack([values, values[:, 0] ** 2])


def build_models():
    """标准化器与模型一起保存，后续验证可在各训练折内独立拟合。"""
    kernel = AdditiveRBF(
        length_scale=[1.0, 1.0], length_scale_bounds=(0.35, 10.0)
    ) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-4, 10.0))
    return {
        "gpr": make_pipeline(StandardScaler(), GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, alpha=1e-8,
            n_restarts_optimizer=2, random_state=RANDOM_SEED,
        )),
        "linear": make_pipeline(StandardScaler(), LinearRegression()),
        "quadratic": make_pipeline(
            StandardScaler(), FunctionTransformer(additive_quadratic), LinearRegression()
        ),
    }
