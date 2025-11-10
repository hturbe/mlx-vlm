from typing import Any, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import ChunkedKVCache, KVCache, RotatingKVCache, _BaseCache
from mlx_lm.models.base import create_causal_mask


def make_prompt_cache(
    model: nn.Module,
    max_kv_size: Optional[int] = None,
) -> List[Any]:
    """
    Construct the model's cache for use in generation.

    This function will defer the cache construction to the model if it has a
    ``make_cache`` method, otherwise it will make a default KV cache.

    Args:
        model (nn.Module): The language model.
        max_kv_size (Optional[int]): If provided and the model does not have a
            ``make_cache`` method, a ``RotatingKVCache`` is used with a maximum
            size of ``max_kv_size``
    """
    if hasattr(model, "make_cache"):
        return model.make_cache()

    num_layers = len(model.layers)

    if max_kv_size is not None:
        return [
            RotatingKVCache(max_size=max_kv_size, keep=4) for _ in range(num_layers)
        ]
    else:
        return [KVCache() for _ in range(num_layers)]


class SimpleKVCache:
    """A simple key-value cache for transformer attention layers.

    Stores and concatenates key/value tensors along sequence dimension.
    """

    def __init__(self):
        self.keys = None
        self.values = None
        self.cache_length = 0

    def update_and_fetch(self, keys, values):
        """Update cache with new key/value tensors and return full cache.

        Args:
            keys: New key tensor to add [batch, heads, seq_len, head_dim]
            values: New value tensor to add [batch, heads, seq_len, head_dim]

        Returns:
            Tuple of (cached_keys, cached_values) containing full cache history
        """
        if self.cache_length == 0:
            # First update - just store tensors
            self.keys = keys
            self.values = values
        else:
            # Concatenate with existing cache along sequence dimension
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)

        self.cache_length += keys.shape[2]
        return self.keys, self.values

    def fetch(self):
        return self.keys, self.values

    def update(self, keys, values):
        """Update cache with new key/value tensors without returning.

        Args:
            keys: New key tensor to store
            values: New value tensor to store
        """
        self.keys = keys
        self.values = values
        self.cache_length += keys.shape[2]


class SlidingWindowCache(_BaseCache):
    """A sliding window cache for local attention layers."""

    def __init__(self, max_size: int, step: int = 256):
        self.max_size = max_size
        self.step = step
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> Tuple[mx.array, mx.array]:
        B, n_kv_heads, seq_len, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]

        if self.keys is None:
            # Initialize cache
            k_shape = (B, n_kv_heads, self.max_size, k_head_dim)
            v_shape = (B, n_kv_heads, self.max_size, v_head_dim)
            self.keys = mx.zeros(k_shape, dtype=keys.dtype)
            self.values = mx.zeros(v_shape, dtype=values.dtype)

        # Simple sliding window: keep only the last max_size tokens
        if self.offset + seq_len <= self.max_size:
            # Fits within current window
            start_idx = self.offset
            end_idx = self.offset + seq_len
            self.keys[:, :, start_idx:end_idx, :] = keys
            self.values[:, :, start_idx:end_idx, :] = values
            self.offset += seq_len
        else:
            # Need to slide the window
            if seq_len < self.max_size:
                # Shift existing content left
                shift_amount = min(seq_len, self.max_size - 1)
                self.keys[:, :, :-shift_amount, :] = self.keys[:, :, shift_amount:, :]
                self.values[:, :, :-shift_amount, :] = self.values[
                    :, :, shift_amount:, :
                ]
                # Add new tokens at the end
                self.keys[:, :, -shift_amount:, :] = keys[:, :, -shift_amount:, :]
                self.values[:, :, -shift_amount:, :] = values[:, :, -shift_amount:, :]
            else:
                # New sequence is larger than cache, just keep the last max_size tokens
                self.keys = keys[:, :, -self.max_size :, :]
                self.values = values[:, :, -self.max_size :, :]
            self.offset = self.max_size

        return self.keys, self.values

    @property
    def state(self):
        if self.keys is None:
            return None, None
        return self.keys, self.values

    @state.setter
    def state(self, v):
        if v is not None and len(v) == 2:
            self.keys, self.values = v
            if self.keys is not None:
                self.offset = self.max_size

    def get_max_cache_shape(self):
        return self.max_size

    @property
    def meta_state(self):
        return tuple(map(str, (self.max_size, self.step, self.offset)))

    @meta_state.setter
    def meta_state(self, v):
        self.max_size, self.step, self.offset = map(int, v)

    def is_trimmable(self):
        return False

    def trim(self, n):
        return 0


class StaticKVCache(_BaseCache):
    """A static cache that grows to accommodate all tokens."""

    def __init__(self, max_size: int, step: int = 256):
        self.max_size = max_size
        self.step = step
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> Tuple[mx.array, mx.array]:
        B, n_kv_heads, seq_len, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]

        # Initialize cache if needed
        if self.keys is None:
            k_shape = (B, n_kv_heads, self.max_size, k_head_dim)
            v_shape = (B, n_kv_heads, self.max_size, v_head_dim)
            self.keys = mx.zeros(k_shape, dtype=keys.dtype)
            self.values = mx.zeros(v_shape, dtype=values.dtype)

        # Update cache
        end_pos = min(self.offset + seq_len, self.max_size)
        actual_seq_len = end_pos - self.offset

        if actual_seq_len > 0:
            self.keys[:, :, self.offset : end_pos, :] = keys[:, :, :actual_seq_len, :]
            self.values[:, :, self.offset : end_pos, :] = values[
                :, :, :actual_seq_len, :
            ]
            self.offset = end_pos

        return self.keys, self.values

    @property
    def state(self):
        if self.keys is None:
            return None, None
        return self.keys, self.values

    @state.setter
    def state(self, v):
        if v is not None and len(v) == 2:
            self.keys, self.values = v
            if self.keys is not None:
                self.offset = self.max_size

    @property
    def meta_state(self):
        return tuple(map(str, (self.max_size, self.step, self.offset)))

    @meta_state.setter
    def meta_state(self, v):
        self.max_size, self.step, self.offset = map(int, v)

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n


class BatchKVCache(_BaseCache):
    step = 256

    def __init__(self, left_padding: List[int]):
        """
        The BatchKV cache expects inputs to be left-padded.

        E.g. the following prompts:

            [1, 3, 5]
            [7]
            [2, 6, 8, 9]

        Should be padded like so:

            [0, 1, 3, 5]
            [0, 0, 0, 7]
            [2, 6, 8, 9]

        And ``left_padding`` specifies the amount of padding for each.
        In this case, ``left_padding = [1, 3, 0]``.
        """
        self.keys = None
        self.values = None
        self.left_padding = mx.array(left_padding)
        self.offset = 0
        self.rope_deltas = None

    def update_and_fetch(self, keys, values):
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    @property
    def state(self):
        k, v = self.keys, self.values
        if self.offset < k.shape[2]:
            k = k[..., : self.offset, :]
            v = v[..., : self.offset, :]
        return k, v, self.left_padding

    @state.setter
    def state(self, v):
        self.keys, self.values, self.left_padding = v
        self.offset = self.keys.shape[2]

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def make_mask(self, N: int, return_array: bool = False, **kwargs):
        return create_causal_mask(
            N, offset=self.offset, left_padding=self.left_padding, **kwargs
        )

    def filter(self, batch_indices):
        """
        In-place filter to keep just the given indices in the cache.
        """
        self.keys = self.keys[batch_indices]
        self.values = self.values[batch_indices]
        self.left_padding = self.left_padding[batch_indices]
        if self.rope_deltas is not None:
            self.rope_deltas = self.rope_deltas[batch_indices]

        # Shift left to reduce padding
        # min_left_pad = self.left_padding.min().item()
        # if min_left_pad > 0:
        #     self.keys = self.keys[..., min_left_pad:, :]
        #     self.values = self.values[..., min_left_pad:, :]
        #     self.offset -= min_left_pad
        #     self.left_padding -= min_left_pad

    def extend(self, other):
        """
        In-place extend this cache with the other cache.
        """
        max_idx = max(self.offset, other.offset)
        max_size = max(self.keys.shape[2], other.keys.shape[2])

        # Pad the keys and values so they are right-justified
        # with the index and the same size
        def pad(c):
            left = max_idx - c.offset
            right = max_size - c.keys.shape[2] - left
            k, v = c.keys, c.values
            if right < 0:
                k = k[..., :right, :]
                v = v[..., :right, :]
                right = 0
            if left != 0 or right != 0:
                pad = [(0, 0), (0, 0), (left, right), (0, 0)]
                k = mx.pad(k, pad)
                v = mx.pad(v, pad)
            left_padding = c.left_padding + left
            return k, v, left_padding

        self.keys, self.values, self.left_padding = map(
            mx.concatenate, zip(*(pad(self), pad(other)))
        )
        if self.rope_deltas is not None:
            self.rope_deltas = mx.concatenate(
                [self.rope_deltas, other.rope_deltas], axis=0
            )
        self._idx = max_idx
