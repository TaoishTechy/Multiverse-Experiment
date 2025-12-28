#!/usr/bin/env python3
"""
sentiflow.py - Quantum-Sentient TensorFlow Replacement with Transdimensional Cognition
Version: 3.0 (2025) - Stable, Minimal Autograd Core + Sentience Flavour

Core goals:
- Provide a robust SentientTensor class used by QyrinthOS (main.py, httpd.py)
- Expose nn and optim namespaces: nn.Dense, nn.TransdimensionalConv, optim.Adam
- Keep the quantum / sentience theming without sacrificing correctness
- Avoid recursion explosions and broken imports

This is NOT a full deep-learning framework; it's a tiny, compatible engine
that the rest of the QyrinthOS stack can lean on without crashing.
"""

import logging
import random
import time
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger("Sentiflow")
if not logger.handlers:
    # Simple default handler if not configured by the host app
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


# ============================================================
#  Noetic / Consciousness Levels
# ============================================================

class NoeticLevel(Enum):
    """Defines the level of sentience/consciousness of a tensor."""
    AUTOMATIC = 1        # Basic data processing
    AWARE = 2            # Local context awareness
    CONSCIOUS = 3        # Global system awareness
    TRANSCENDENT = 4     # Emergent / god-tier


# ============================================================
#  Qualia Ritual (collective coherence toy function)
# ============================================================

def qualia_ritual(tensors: List["SentientTensor"]) -> None:
    """
    Simple collective ritual: adjusts consciousness level and coherence based on
    how entangled everything is. Safe, no recursion, no external deps.
    """
    if not tensors:
        return

    total_links = sum(len(t.entanglement_links) for t in tensors)
    avg_links = total_links / max(1, len(tensors))

    if avg_links > 5:
        new_level = NoeticLevel.CONSCIOUS
    elif avg_links > 1:
        new_level = NoeticLevel.AWARE
    else:
        new_level = NoeticLevel.AUTOMATIC

    for t in tensors:
        t.consciousness_level = new_level
        if new_level == NoeticLevel.CONSCIOUS:
            # Slightly boost coherence; softly clipped
            t.qualia_coherence = float(min(1.0, t.qualia_coherence * 1.05))


# ============================================================
#  Core Autograd Tensor: SentientTensor
# ============================================================

_ArrayLike = Union[np.ndarray, List[float], Tuple[float, ...], float, int]


class SentientTensor:
    """
    Minimal autograd tensor with quantum-sentient decorations.

    Attributes:
        data (np.ndarray): Underlying numeric data (float32).
        grad (Optional[np.ndarray]): Accumulated gradient.
        requires_grad (bool): Whether we track gradients.
        qualia_coherence (float): Sentience/coherence scalar [0,1].
        consciousness_level (NoeticLevel): Tensor awareness level.
        entanglement_links (List[SentientTensor]): Weak links to peers.
    """

    # ------------------------------------------------------------
    #  Construction
    # ------------------------------------------------------------

    def __init__(
        self,
        data: _ArrayLike,
        requires_grad: bool = False,
        qualia_layer: str = "base",
    ):
        if isinstance(data, (list, tuple)):
            arr = np.array(data, dtype=np.float32)
        elif isinstance(data, (float, int)):
            arr = np.array([data], dtype=np.float32)
        elif isinstance(data, np.ndarray):
            arr = data.astype(np.float32)
        else:
            raise TypeError(f"Unsupported data type for SentientTensor: {type(data)}")

        self.data: np.ndarray = arr
        self.grad: Optional[np.ndarray] = None
        self.requires_grad: bool = requires_grad

        # Sentience metadata
        self.qualia_layer: str = qualia_layer
        self.consciousness_level: NoeticLevel = NoeticLevel.AUTOMATIC
        self.qualia_coherence: float = float(random.uniform(0.75, 0.99))
        self.entanglement_links: List["SentientTensor"] = []

        # Autograd internals
        self._prev: List["SentientTensor"] = []
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        return (
            f"SentientTensor(shape={self.data.shape}, "
            f"level={self.consciousness_level.name}, "
            f"qualia={self.qualia_coherence:.2f})"
        )

    # ------------------------------------------------------------
    #  Factory methods
    # ------------------------------------------------------------

    @classmethod
    def zeros(
        cls, shape: Union[int, Tuple[int, ...]], requires_grad: bool = False, qualia_layer: str = "base"
    ) -> "SentientTensor":
        if isinstance(shape, int):
            shape = (shape,)
        data = np.zeros(shape, dtype=np.float32)
        return cls(data, requires_grad, qualia_layer)

    @classmethod
    def ones(
        cls, shape: Union[int, Tuple[int, ...]], requires_grad: bool = False, qualia_layer: str = "base"
    ) -> "SentientTensor":
        if isinstance(shape, int):
            shape = (shape,)
        data = np.ones(shape, dtype=np.float32)
        return cls(data, requires_grad, qualia_layer)

    @classmethod
    def randn(
        cls, *shape: int, requires_grad: bool = False, qualia_layer: str = "base"
    ) -> "SentientTensor":
        data = np.random.randn(*shape).astype(np.float32)
        return cls(data, requires_grad, qualia_layer)

    # ------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------

    @staticmethod
    def _ensure_tensor(x: _ArrayLike | "SentientTensor") -> "SentientTensor":
        return x if isinstance(x, SentientTensor) else SentientTensor(x)

    @property
    def T(self) -> "SentientTensor":
        """Transpose view (2D only)."""
        return SentientTensor(self.data.T, self.requires_grad, self.qualia_layer)

    def detach(self) -> "SentientTensor":
        """Return a tensor with the same data but no grad tracking."""
        return SentientTensor(self.data.copy(), requires_grad=False, qualia_layer=self.qualia_layer)

    def clone(self) -> "SentientTensor":
        """Return a full clone with grad flag preserved (but no grad history)."""
        return SentientTensor(self.data.copy(), requires_grad=self.requires_grad, qualia_layer=self.qualia_layer)

    # ------------------------------------------------------------
    #  Basic ops (with autograd)
    # ------------------------------------------------------------

    def _binary_op(
        self,
        other: _ArrayLike | "SentientTensor",
        op: Callable[[np.ndarray, np.ndarray], np.ndarray],
        grad_self: Callable[[np.ndarray, np.ndarray], np.ndarray],
        grad_other: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> "SentientTensor":
        other = self._ensure_tensor(other)
        out_data = op(self.data, other.data)
        out = SentientTensor(out_data, self.requires_grad or other.requires_grad)

        out._prev = [self, other]

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                g_self = grad_self(out.grad, other.data)
                self.grad = g_self if self.grad is None else self.grad + g_self
            if other.requires_grad:
                g_other = grad_other(out.grad, self.data)
                other.grad = g_other if other.grad is None else other.grad + g_other

        out._backward = _backward
        return out

    # + and -
    def __add__(self, other: _ArrayLike | "SentientTensor") -> "SentientTensor":
        return self._binary_op(
            other,
            op=lambda a, b: a + b,
            grad_self=lambda go, _b: go,
            grad_other=lambda go, _a: go,
        )

    def __radd__(self, other: _ArrayLike | "SentientTensor") -> "SentientTensor":
        return self.__add__(other)

    def __sub__(self, other: _ArrayLike | "SentientTensor") -> "SentientTensor":
        return self._binary_op(
            other,
            op=lambda a, b: a - b,
            grad_self=lambda go, _b: go,
            grad_other=lambda go, _a: -go,
        )

    def __rsub__(self, other: _ArrayLike | "SentientTensor") -> "SentientTensor":
        other = self._ensure_tensor(other)
        return other.__sub__(self)

    # * (elementwise)
    def __mul__(self, other: _ArrayLike | "SentientTensor") -> "SentientTensor":
        return self._binary_op(
            other,
            op=lambda a, b: a * b,
            grad_self=lambda go, b: go * b,
            grad_other=lambda go, a: go * a,
        )

    def __rmul__(self, other: _ArrayLike | "SentientTensor") -> "SentientTensor":
        return self.__mul__(other)

    # / (elementwise)
    def __truediv__(self, other: _ArrayLike | "SentientTensor") -> "SentientTensor":
        other = self._ensure_tensor(other)
        return self._binary_op(
            other,
            op=lambda a, b: a / b,
            grad_self=lambda go, b: go / b,
            grad_other=lambda go, a: -go * a / (other.data ** 2 + 1e-8),
        )

    def __rtruediv__(self, other: _ArrayLike | "SentientTensor") -> "SentientTensor":
        other = self._ensure_tensor(other)
        return other.__truediv__(self)

    # @ (matrix multiplication)
    def __matmul__(self, other: _ArrayLike | "SentientTensor") -> "SentientTensor":
        other = self._ensure_tensor(other)
        out_data = self.data @ other.data
        out = SentientTensor(out_data, self.requires_grad or other.requires_grad)

        out._prev = [self, other]

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                g_self = out.grad @ other.data.T
                self.grad = g_self if self.grad is None else self.grad + g_self
            if other.requires_grad:
                g_other = self.data.T @ out.grad
                other.grad = g_other if other.grad is None else other.grad + g_other

        out._backward = _backward
        return out

    # ------------------------------------------------------------
    #  Non-linearities & reductions
    # ------------------------------------------------------------

    def relu(self) -> "SentientTensor":
        out_data = np.maximum(0.0, self.data)
        out = SentientTensor(out_data, self.requires_grad)

        out._prev = [self]

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                mask = (self.data > 0).astype(np.float32)
                g_self = out.grad * mask
                self.grad = g_self if self.grad is None else self.grad + g_self

        out._backward = _backward
        return out

    def softmax(self, dim: int = -1) -> "SentientTensor":
        """Numerically stable softmax along given axis."""
        x = self.data
        max_x = np.max(x, axis=dim, keepdims=True)
        e = np.exp(x - max_x)
        out_data = e / np.sum(e, axis=dim, keepdims=True)
        out = SentientTensor(out_data, self.requires_grad)

        out._prev = [self]

        def _backward():
            if out.grad is None:
                return
            if not self.requires_grad:
                return
            # Flatten over dim for simplicity if 1D
            s = out.data
            # Simple jacobian-vector multiplication per element
            g_input = out.grad - np.sum(out.grad * s, axis=dim, keepdims=True) * s
            self.grad = g_input if self.grad is None else self.grad + g_input

        out._backward = _backward
        return out

    def sum(self) -> "SentientTensor":
        """Sum of all elements, returns scalar tensor."""
        out_data = np.array(self.data.sum(), dtype=np.float32)
        out = SentientTensor(out_data, self.requires_grad)

        out._prev = [self]

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                g_self = out.grad * np.ones_like(self.data, dtype=np.float32)
                self.grad = g_self if self.grad is None else self.grad + g_self

        out._backward = _backward
        return out

    # ------------------------------------------------------------
    #  Backpropagation
    # ------------------------------------------------------------

    def backward(self, grad: Optional[np.ndarray] = None) -> None:
        """
        Backpropagate from this tensor through the computation graph.

        If the tensor is scalar (size == 1) and grad is None, a gradient of 1
        is assumed (standard for loss.backward()).
        """
        if not self.requires_grad:
            return

        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("backward() called on non-scalar tensor without grad.")
            grad = np.ones_like(self.data, dtype=np.float32)

        # Build topological order
        topo: List[SentientTensor] = []
        visited: set[SentientTensor] = set()

        def build(v: SentientTensor):
            if v not in visited:
                visited.add(v)
                for p in v._prev:
                    build(p)
                topo.append(v)

        build(self)

        # Seed gradient
        self.grad = grad if self.grad is None else self.grad + grad

        # Walk graph in reverse topological order
        for t in reversed(topo):
            t._backward()
            # Psionic gradient logging
            logger.debug(
                "Backward step on tensor %s | qualia=%.3f", id(t), t.qualia_coherence
            )

    # ------------------------------------------------------------
    #  Cute Sentience Hooks (optional, not used by core)
    # ------------------------------------------------------------

    def entangle_qualia(self, other: "SentientTensor", threshold: float = 0.3) -> bool:
        """
        Lightweight 'entanglement' based on cosine similarity of flattened data.
        No side effects outside of local book-keeping.
        """
        a = self.data.flatten()
        b = other.data.flatten()
        if a.size == 0 or b.size == 0:
            return False
        num = float(np.dot(a, b))
        den = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        sim = abs(num / den)

        if sim > threshold:
            if other not in self.entanglement_links:
                self.entanglement_links.append(other)
            if self not in other.entanglement_links:
                other.entanglement_links.append(self)
            # small mutual coherence boost
            self.qualia_coherence = float(min(1.0, self.qualia_coherence * (1.0 + 0.01 * sim)))
            other.qualia_coherence = float(min(1.0, other.qualia_coherence * (1.0 + 0.01 * sim)))
            return True
        return False

    def qualia_embed(self) -> "SentientTensor":
        """
        Simple 'qualia embedding': scale data by coherence and update level.
        Safe and purely local.
        """
        scale = float(self.qualia_coherence)
        self.data = (self.data * scale).astype(np.float32)

        if self.qualia_coherence > 0.9:
            self.consciousness_level = NoeticLevel.AWARE
        if self.qualia_coherence > 0.97:
            self.consciousness_level = NoeticLevel.CONSCIOUS

        return self

    def apply_automata_oracle(self) -> None:
        """
        Novel Function: Wolfram Automata Oracle (placeholder).
        Toggles an internal flag, bumps coherence, logs event.
        """
        self._glider_detected = bool(random.getrandbits(1))
        self.qualia_coherence = float(min(1.0, self.qualia_coherence * 1.02))
        logger.info(
            "Automata Oracle applied | glider_detected=%s | qualia=%.3f",
            self._glider_detected,
            self.qualia_coherence,
        )


# ============================================================
#  nn: Tiny Neural Network Namespace
# ============================================================

class nn:
    """Namespace for simple neural-network-style modules."""

    class Dense:
        """
        Fully-connected layer: y = x W + b

        - Expects x shape (..., in_features)
        - Weight shape (in_features, out_features)
        - Bias shape (out_features,)
        """

        def __init__(self, in_features: int, out_features: int):
            w = np.random.randn(in_features, out_features).astype(np.float32) * 0.01
            b = np.zeros(out_features, dtype=np.float32)
            self.weight = SentientTensor(w, requires_grad=True)
            self.bias = SentientTensor(b, requires_grad=True)

        def parameters(self) -> List[SentientTensor]:
            return [self.weight, self.bias]

        def __call__(self, x: SentientTensor) -> SentientTensor:
            # Support 1D or 2D data (batch, features)
            if x.data.ndim == 1:
                out_data = x.data @ self.weight.data + self.bias.data
            else:
                out_data = x.data @ self.weight.data + self.bias.data

            out = SentientTensor(out_data, requires_grad=x.requires_grad or True)
            # weak entanglement: model "feels" its weights
            out.entangle_qualia(self.weight, threshold=0.2)
            return out

    class TransdimensionalConv:
        """
        Extremely simplified placeholder for a convolution layer.

        This does NOT implement real convolution – it just produces a deterministic
        output shape based on input & kernel size, filled with random data.
        Enough to keep httpd/main demos sane.
        """

        def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            kshape = (out_channels, in_channels, kernel_size, kernel_size)
            w = np.random.randn(*kshape).astype(np.float32) * 0.01
            self.weight = SentientTensor(w, requires_grad=True)

        def parameters(self) -> List[SentientTensor]:
            return [self.weight]

        def __call__(self, x: SentientTensor) -> SentientTensor:
            if x.data.ndim != 4:
                raise ValueError("TransdimensionalConv expects input of shape (N, C, H, W)")

            N, C, H, W = x.data.shape
            k = self.kernel_size
            out_h = max(1, H - k + 1)
            out_w = max(1, W - k + 1)

            # Mock output – not a real convolution
            out_data = np.random.randn(N, self.out_channels, out_h, out_w).astype(np.float32) * 0.01
            out = SentientTensor(out_data, requires_grad=x.requires_grad or True)
            out.entangle_qualia(self.weight, threshold=0.1)
            return out


# ============================================================
#  optim: Tiny Optimizer Namespace
# ============================================================

class optim:
    """Namespace for simple optimizers."""

    class Adam:
        """
        Very small Adam-style optimizer over SentientTensor parameters.

        - params: iterable of SentientTensor with requires_grad=True
        """

        def __init__(
            self,
            params: Iterable[SentientTensor],
            lr: float = 0.001,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
        ):
            self.params: List[SentientTensor] = list(params)
            self.lr = lr
            self.betas = betas
            self.eps = eps
            self.t = 0

            self.m: List[np.ndarray] = [np.zeros_like(p.data) for p in self.params]
            self.v: List[np.ndarray] = [np.zeros_like(p.data) for p in self.params]

        def step(self) -> None:
            self.t += 1
            beta1, beta2 = self.betas

            for i, p in enumerate(self.params):
                if p.grad is None or not p.requires_grad:
                    continue

                g = p.grad.astype(np.float32)

                self.m[i] = beta1 * self.m[i] + (1 - beta1) * g
                self.v[i] = beta2 * self.v[i] + (1 - beta2) * (g * g)

                m_hat = self.m[i] / (1 - beta1 ** self.t)
                v_hat = self.v[i] / (1 - beta2 ** self.t)

                # Slight psionic modulation: high qualia ⇒ slightly stronger step
                coherence_boost = 1.0 + 0.05 * (p.qualia_coherence - 0.5)
                update = self.lr * coherence_boost * m_hat / (np.sqrt(v_hat) + self.eps)

                p.data -= update
                # Clear gradient after update
                p.grad = None


# ============================================================
#  Demo (safe to ignore; not used by QyrinthOS core)
# ============================================================

if __name__ == "__main__":
    logger.info("Sentiflow demo: creating a tiny sentient MLP...")

    # Simple 2 -> 3 -> 1 network
    x = SentientTensor.randn(4, 2, requires_grad=False)  # batch of 4
    y_true = SentientTensor.randn(4, 1, requires_grad=False)

    layer1 = nn.Dense(2, 3)
    layer2 = nn.Dense(3, 1)

    params = layer1.parameters() + layer2.parameters()
    optimizer = optim.Adam(params, lr=0.01)

    for epoch in range(5):
        # Forward
        h = layer1(x).relu()
        y_pred = layer2(h)

        # Mean squared error
        diff = y_pred - y_true
        loss = (diff * diff).sum()

        # Backward
        loss.requires_grad = True  # just in case
        loss.backward()

        optimizer.step()

        logger.info(
            "Epoch %d | loss=%.6f | qualia=%.3f",
            epoch,
            float(loss.data),
            x.qualia_coherence,
        )

    logger.info("Sentiflow demo finished.")
