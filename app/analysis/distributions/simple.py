import numpy as np
from pydantic import BaseModel, Field


class Constant:
    def __init__(self, value: float) -> None:
        self.value = value

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.full(n, self.value)

    def point(self) -> float:
        return self.value


class GammaFromMeanSD(BaseModel):
    mean: float
    sd: float

    def sample(self, n: int, rng: np.random.Generator):
        k = (self.mean / self.sd) ** 2
        theta = (self.sd**2) / self.mean
        return rng.gamma(shape=k, scale=theta, size=n)

    def point(self) -> float:
        return self.mean


class GammaFromMeanCV(BaseModel):
    mean: float
    cv: float

    def sample(self, n: int, rng: np.random.Generator):
        if self.cv <= 0:
            raise ValueError("CV must be > 0")

        shape = 1 / (self.cv**2)
        scale = self.mean * (self.cv**2)

        return rng.gamma(shape=shape, scale=scale, size=n)

    def point(self) -> float:
        return self.mean


class BetaFromMeanSD(BaseModel):
    mean: float = Field(..., gt=0, lt=1)
    sd: float | None = Field(default=None, gt=0)
    cv: float | None = Field(default=None, ge=0)

    def _resolve_sd(self) -> float:
        if self.sd is not None:
            return self.sd

        if self.cv is not None:
            return self.mean * self.cv

        raise ValueError("Either sd or cv must be provided.")

    def _to_beta_params(self):
        sd = self._resolve_sd()
        var = sd**2

        max_var = self.mean * (1 - self.mean)
        if var >= max_var:
            raise ValueError(
                f"Invalid Beta parameters: mean={self.mean}, sd={sd}. "
                f"Variance must be < mean*(1-mean)={max_var:.6f}"
            )

        common = (self.mean * (1 - self.mean) / var) - 1

        alpha = self.mean * common
        beta = (1 - self.mean) * common

        alpha = max(alpha, 1e-8)
        beta = max(beta, 1e-8)

        return alpha, beta

    def sample(self, n: int, rng: np.random.Generator):
        alpha, beta = self._to_beta_params()
        return rng.beta(alpha, beta, size=n)

    def point(self) -> float:
        return self.mean


class BetaDist(BaseModel):
    """Beta distribution parameterized directly by evidence counts."""

    alpha: float = Field(..., gt=0)
    beta: float = Field(..., gt=0)

    def sample(self, n: int, rng: np.random.Generator):
        return rng.beta(self.alpha, self.beta, size=n)

    def point(self) -> float:
        return self.alpha / (self.alpha + self.beta)


class TriangularDist(BaseModel):
    left: float
    mode: float
    right: float

    def sample(self, n: int, rng: np.random.Generator):
        return rng.triangular(self.left, self.mode, self.right, size=n)

    def point(self) -> float:
        return (self.left + self.mode + self.right) / 3
