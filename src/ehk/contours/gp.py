"""Small exact Matérn GP with known noise, censoring, and fidelity input."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import ndtr, ndtri

from .protocol import ContourProtocol
from .store import Evaluation

FloatArray = NDArray[np.float64]


def _matern52(a: FloatArray, b: FloatArray, length_scales: FloatArray) -> FloatArray:
    delta = (a[:, None, :] - b[None, :, :]) / length_scales
    distance = np.sqrt(np.sum(delta * delta, axis=2))
    scaled = math.sqrt(5.0) * distance
    return (1.0 + scaled + scaled * scaled / 3.0) * np.exp(-scaled)


@dataclass(frozen=True)
class Prediction:
    mean: FloatArray
    variance: FloatArray


class MaternGP:
    """Exact ARD Matérn-5/2 GP over coordinates plus inverse fidelity.

    Censoring uses an iterative truncated-normal conditional expectation.  The
    censored value is never treated as an exact observation; its effective
    variance is enlarged at each iteration.
    """

    def __init__(self, protocol: ContourProtocol):
        self.protocol = protocol
        self.feature_dimension = protocol.dimension + 1
        self.variance_calibration_scale = 1.0
        self.loo_standardized_rmse = math.nan
        self._fitted = False

    def _features(self, x: Sequence[Sequence[float]], fidelity: Sequence[int]) -> FloatArray:
        values = np.asarray(x, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.protocol.dimension:
            raise ValueError("x has wrong shape")
        domain = np.asarray(self.protocol.domain, dtype=float)
        normalized = (values - domain[:, 0]) / (domain[:, 1] - domain[:, 0])
        inverse = 1.0 / np.asarray(fidelity, dtype=float)
        fidelity_values = np.asarray(self.protocol.fidelities, dtype=float)
        inverse_min = 1.0 / fidelity_values[-1]
        inverse_max = 1.0 / fidelity_values[0]
        if math.isclose(inverse_min, inverse_max):
            level = np.zeros_like(inverse)
        else:
            level = (inverse - inverse_min) / (inverse_max - inverse_min)
        return np.column_stack((normalized, level))

    @staticmethod
    def _training_rows(evaluations: Sequence[Evaluation]) -> list[Evaluation]:
        return [
            item for item in evaluations
            if item.status == "ok" and item.role != "validation"
            and (item.value is not None or item.censor_bound is not None)
        ]

    def fit(self, evaluations: Sequence[Evaluation]) -> MaternGP:
        rows = self._training_rows(evaluations)
        if len(rows) < 2:
            raise ValueError("at least two successful training evaluations are required")
        self.features = self._features([item.x for item in rows], [item.fidelity for item in rows])
        raw = np.asarray([
            item.value if item.censoring == "none" else item.censor_bound
            for item in rows
        ], dtype=float)
        self.y_location = float(np.mean(raw))
        self.y_scale = max(float(np.std(raw)), 1e-8)
        observed = (raw - self.y_location) / self.y_scale
        known_noise = np.asarray([item.noise_variance for item in rows], dtype=float) / self.y_scale**2
        censoring = np.asarray([item.censoring for item in rows], dtype=object)
        bounds = observed.copy()

        initial = np.concatenate((
            np.full(self.feature_dimension, math.log(0.35)),
            [0.0, math.log(1e-6)],
        ))
        limits = [(-4.5, 2.0)] * self.feature_dimension + [(-5.0, 5.0), (-18.0, -1.0)]
        pseudo = observed.copy()
        effective_noise = known_noise.copy()

        for _ in range(4):
            result = minimize(
                self._negative_log_likelihood,
                initial,
                args=(pseudo, effective_noise),
                method="L-BFGS-B",
                bounds=limits,
            )
            theta = result.x if np.all(np.isfinite(result.x)) else initial
            self._set_posterior(theta, pseudo, effective_noise)
            prediction = self._predict_features(self.features, include_noise=False)
            standard = np.sqrt(np.maximum(prediction.variance + known_noise, 1e-12))
            for index, kind in enumerate(censoring):
                if kind == "none":
                    continue
                z = (bounds[index] - prediction.mean[index]) / standard[index]
                if kind == "left":
                    probability = max(float(ndtr(z)), 1e-12)
                    ratio = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi) / probability
                    pseudo[index] = prediction.mean[index] - standard[index] * ratio
                else:
                    probability = max(float(ndtr(-z)), 1e-12)
                    ratio = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi) / probability
                    pseudo[index] = prediction.mean[index] + standard[index] * ratio
                effective_noise[index] = max(known_noise[index], standard[index] ** 2)
            initial = theta

        self._set_posterior(initial, pseudo, effective_noise)
        self._calibrate_predictive_variance()
        self.rows = tuple(rows)
        self._fitted = True
        return self

    def _calibrate_predictive_variance(self) -> None:
        """Inflate predictive variance from training-only leave-one-out residuals.

        The nugget handles unresolved local structure in the mean fit.  This
        optional scale calibration prevents a deterministic but imperfect
        stationary surrogate from reporting spuriously narrow intervals.  It
        never shrinks posterior variance and does not inspect validation rows.
        """

        mode = str(self.protocol.design.get("variance_calibration", "none"))
        if mode == "none":
            return
        if mode != "loo_quantile":
            raise ValueError(f"unsupported variance_calibration: {mode}")
        quantile = float(
            self.protocol.design.get("variance_calibration_quantile", 0.95)
        )
        if not 0.5 < quantile < 1.0:
            raise ValueError("variance_calibration_quantile must be between 0.5 and 1")
        precision_diagonal = np.diag(
            cho_solve(
                self.factor,
                np.eye(len(self.features), dtype=float),
                check_finite=False,
            )
        )
        standardized = self.alpha / np.sqrt(np.maximum(precision_diagonal, 1e-12))
        finite = np.abs(standardized[np.isfinite(standardized)])
        if finite.size < 3:
            return
        self.loo_standardized_rmse = float(np.sqrt(np.mean(finite**2)))
        empirical = float(np.quantile(finite, quantile, method="higher"))
        gaussian = float(ndtri(0.5 + quantile / 2.0))
        self.variance_calibration_scale = max(1.0, empirical / gaussian)

    def _kernel(self, a: FloatArray, b: FloatArray, theta: FloatArray) -> FloatArray:
        lengths = np.exp(theta[:self.feature_dimension])
        amplitude = math.exp(theta[self.feature_dimension])
        return amplitude * _matern52(a, b, lengths)

    def _negative_log_likelihood(
        self,
        theta: FloatArray,
        y: FloatArray,
        known_noise: FloatArray,
    ) -> float:
        kernel = self._kernel(self.features, self.features, theta)
        nugget = math.exp(theta[-1])
        kernel.flat[::len(kernel) + 1] += known_noise + nugget + 1e-10
        try:
            factor = cho_factor(kernel, lower=True, check_finite=False)
            alpha = cho_solve(factor, y, check_finite=False)
        except np.linalg.LinAlgError:
            return 1e100
        log_det = 2 * np.sum(np.log(np.diag(factor[0])))
        return float(0.5 * y @ alpha + 0.5 * log_det + 0.5 * len(y) * math.log(2 * math.pi))

    def _set_posterior(self, theta: FloatArray, y: FloatArray, known_noise: FloatArray) -> None:
        kernel = self._kernel(self.features, self.features, theta)
        kernel.flat[::len(kernel) + 1] += known_noise + math.exp(theta[-1]) + 1e-10
        self.theta = np.asarray(theta, dtype=float)
        self.factor = cho_factor(kernel, lower=True, check_finite=False)
        self.alpha = cho_solve(self.factor, y, check_finite=False)

    def _predict_features(self, features: FloatArray, *, include_noise: bool) -> Prediction:
        cross = self._kernel(features, self.features, self.theta)
        mean = cross @ self.alpha
        solved = cho_solve(self.factor, cross.T, check_finite=False)
        amplitude = math.exp(self.theta[self.feature_dimension])
        variance = amplitude - np.sum(cross * solved.T, axis=1)
        if include_noise:
            variance += math.exp(self.theta[-1])
        return Prediction(mean, np.maximum(variance, 1e-12))

    def predict(
        self,
        x: Sequence[Sequence[float]],
        fidelity: Sequence[int],
        *,
        include_noise: bool = False,
    ) -> Prediction:
        if not self._fitted:
            raise RuntimeError("GP is not fitted")
        normalized = self._predict_features(self._features(x, fidelity), include_noise=include_noise)
        return Prediction(
            normalized.mean * self.y_scale + self.y_location,
            normalized.variance
            * self.y_scale**2
            * self.variance_calibration_scale**2,
        )

    def posterior_covariance(
        self,
        x_a: Sequence[Sequence[float]],
        fidelity_a: Sequence[int],
        x_b: Sequence[Sequence[float]],
        fidelity_b: Sequence[int],
    ) -> FloatArray:
        if not self._fitted:
            raise RuntimeError("GP is not fitted")
        a = self._features(x_a, fidelity_a)
        b = self._features(x_b, fidelity_b)
        prior = self._kernel(a, b, self.theta)
        cross_a = self._kernel(a, self.features, self.theta)
        cross_b = self._kernel(self.features, b, self.theta)
        conditional = prior - cross_a @ cho_solve(self.factor, cross_b, check_finite=False)
        return (
            conditional
            * self.y_scale**2
            * self.variance_calibration_scale**2
        )
