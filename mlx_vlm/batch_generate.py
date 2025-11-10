import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.utils import prepare_inputs
from mlx_vlm.generate import wired_limit
from PIL import Image
from transformers import PreTrainedTokenizer

from .models.cache import BatchKVCache

# from mlx_lm.models.cache import BatchKVCache
# A stream on the default device just for generation
generation_stream = mx.new_stream(mx.default_device())


@dataclass
class BatchStats:
    """
    An data object to hold generation stats.

    Args:
        prompt_tokens (int): The number of prompt tokens processed.
        prompt_tps (float): The prompt processing tokens-per-second.
        prompt_time (float): The time in seconds spent in prompt processing.
        generation_tokens (int): The number of generated tokens.
        generation_tps (float): The tokens-per-second for generation.
        generation_time (float): The time in seconds spent in generation .
        peak_memory (float): The peak memory used so far in GB.
    """

    prompt_tokens: int = 0
    prompt_tps: float = 0
    prompt_time: float = 0
    generation_tokens: int = 0
    generation_tps: float = 0
    generation_time: float = 0
    peak_memory: float = 0


@dataclass
class BatchResponse:
    """
    An data object to hold a batch generation response.

    Args:
        texts: (List[str]): The generated text for each prompt.
        stats (BatchStats): Statistics about the generation.
    """

    texts: List[str]
    stats: BatchStats


@dataclass
class Batch:
    uids: List[int]
    y: mx.array
    logprobs: mx.array
    max_tokens: List[int]
    num_tokens: List[int]
    cache: List[Any]

    def __len__(self):
        return len(self.uids)

    def filter(self, keep_idx: List[int]):
        self.uids = [self.uids[k] for k in keep_idx]
        self.max_tokens = [self.max_tokens[k] for k in keep_idx]
        self.num_tokens = [self.num_tokens[k] for k in keep_idx]
        keep_idx = mx.array(keep_idx, mx.int32)
        self.y = self.y[keep_idx]
        self.logprobs = self.logprobs[keep_idx]
        for c in self.cache:
            c.filter(keep_idx)

    def extend(self, other):
        self.uids.extend(other.uids)
        self.y = mx.concatenate([self.y, other.y])
        self.logprobs = mx.concatenate([self.logprobs, other.logprobs])
        self.num_tokens.extend(other.num_tokens)
        self.max_tokens.extend(other.max_tokens)
        for c, o in zip(self.cache, other.cache):
            c.extend(o)


def _make_cache(model, left_padding):
    """
    Convert a list of regular caches into their corresponding
    batch-aware caches.
    """

    def to_batch_cache(c):
        if isinstance(c, KVCache):
            return BatchKVCache(left_padding)
        else:
            raise ValueError(f"{type(c)} does not yet support batching")

    if hasattr(model, "make_cache"):
        cache = model.make_cache()
        return [to_batch_cache(c) for c in cache]
    else:
        return [BatchKVCache(left_padding) for _ in model.layers]


class BatchGenerator:
    @dataclass
    class Response:
        uid: int
        token: int
        logprobs: mx.array
        finish_reason: Optional[str]

    def __init__(
        self,
        model,
        max_tokens: int = 128,
        stop_tokens: Optional[set] = None,
        sampler: Optional[Callable[[mx.array], mx.array]] = None,
        completion_batch_size: int = 100,
        prefill_batch_size: int = 10,
        prefill_step_size: int = 2048,
        **kwargs,
    ):
        self.model = model
        self.unprocessed_prompts = []
        self.max_tokens = max_tokens
        self.stop_tokens = stop_tokens or set()
        self.sampler = sampler or (lambda x: mx.argmax(x, axis=-1))
        self.uid_count = 0
        self.prefill_step_size = prefill_step_size
        self.prefill_batch_size = prefill_batch_size
        # self.prefill_batch_size = completion_batch_size
        self.prefill_step_size = prefill_step_size
        self.completion_batch_size = completion_batch_size
        self._stats = BatchStats()

        self.active_batch = None

        if not isinstance(self.stop_tokens, list):
            self.stop_tokens = [self.stop_tokens]

    def insert(self, prompts, max_tokens: Union[List[int], int, None] = None):
        uids = []

        if max_tokens is None or isinstance(max_tokens, int):
            max_tokens = [max_tokens or self.max_tokens] * len(prompts)

        for p, m in zip(prompts, max_tokens):
            self.unprocessed_prompts.append((self.uid_count, p, m))
            uids.append(self.uid_count)
            self.uid_count += 1
        # Sort in ascending order of length
        self.unprocessed_prompts = sorted(
            self.unprocessed_prompts, key=lambda x: len(x[1])
        )
        return uids

    def _process_prompts(self, prompts):
        uids, inputs, max_tokens = zip(*prompts)
        pixel_values = [mx.array(p["pixel_values"][0]) for p in inputs]
        pixel_values = mx.concat(pixel_values, axis=0)
        mask = mx.array([p["attention_mask"][0].tolist() for p in inputs])
        image_grid_thw = mx.array([p["image_grid_thw"][0] for p in inputs])
        self._stats.prompt_tokens += mask.sum().item()
        # left_padding = [max_length - l for l in lengths]
        text_inputs = mx.array([p["input_ids"][0] for p in inputs])
        max_length = text_inputs.shape[1]
        lengths = mask.sum(axis=1).tolist()
        left_padding = [max_length - l for l in lengths]
        # text_inputs = _left_pad_prompts(text_inputs, max_length=max_length)

        prompt_cache = _make_cache(self.model.language_model, left_padding)

        # inputs[0]["image_grid_thw"]
        # {'image_grid_thw': array([[1, 74, 74]], dtype=int64)}
        outputs = self.model(
            text_inputs,
            pixel_values,
            cache=prompt_cache,
            mask=mask,
            image_grid_thw=image_grid_thw,
        )
        # mx.eval([c.state for c in prompt_cache])
        logits = outputs.logits[:, -1, :]
        y = self.sampler(logits)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        # y = self.sampler(logprobs)

        # while inputs.shape[1] > 1:
        #     n_to_process = min(self.prefill_step_size, inputs.shape[1] - 1)
        #     self.model(inputs[:, :n_to_process], cache=prompt_cache)
        #     mx.eval([c.state for c in prompt_cache])
        #     inputs = inputs[:, n_to_process:]
        #     mx.clear_cache()

        # mx.async_eval(y, logprobs)
        mx.clear_cache()

        # Clear intermediate variables to free memory
        del pixel_values, mask, image_grid_thw, text_inputs, outputs, logits

        return Batch(
            list(uids),
            y,
            logprobs,
            list(max_tokens),
            [0] * len(uids),
            prompt_cache,
        )

    def _step(
        self,
        input_tokens: mx.array,
        prompt_cache: List[Any],
    ):
        outputs = self.model.language_model(
            input_tokens,
            cache=prompt_cache,
        )
        # logits = logits[:, -1, :]
        logits = outputs.logits[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        y = self.sampler(logits)
        del outputs, logits

        return y, logprobs

    def stats(self):
        self._stats.prompt_tps = self._stats.prompt_tokens / self._stats.prompt_time
        self._stats.generation_tps = (
            self._stats.generation_tokens / self._stats.generation_time
        )
        self._stats.peak_memory = mx.get_peak_memory() / 1e9
        return self._stats

    def _next(self):
        tic = time.perf_counter()

        prompt_processing = False
        batch = self.active_batch
        num_active = len(batch) if batch else 0
        num_to_add = self.completion_batch_size - num_active
        while num_to_add >= self.prefill_batch_size:
            prompts = self.unprocessed_prompts[: self.prefill_batch_size]
            # Finish processing the last examples of the last batch
            if len(prompts) == 0 and num_active > 0:
                break
            # No more prompts and no more completions, all done
            elif len(prompts) == 0:
                self.active_batch = None
                return []
            # Process prompts
            if batch is not None and not prompt_processing:
                # Finish any active completion tokens
                mx.eval(batch.y, batch.logprobs)
                self._stats.generation_time += time.perf_counter() - tic
                tic = time.perf_counter()

            batch = self._process_prompts(prompts)
            self.unprocessed_prompts = self.unprocessed_prompts[
                self.prefill_batch_size :
            ]
            prompt_processing = True
            # If there was no active batch, set it
            if self.active_batch is None:
                self.active_batch = batch
            else:
                self.active_batch.extend(batch)

            num_active = len(self.active_batch)
            num_to_add -= len(batch)

        batch = self.active_batch
        y, logprobs = batch.y, batch.logprobs
        batch.y, batch.logprobs = self._step(y[:, None], batch.cache)
        mx.async_eval(batch.y, batch.logprobs)

        y = y.tolist()
        toc = time.perf_counter()
        if prompt_processing:
            self._stats.prompt_time += toc - tic
        else:
            self._stats.generation_time += toc - tic
        keep_idx = []
        end_idx = []
        responses = []
        for e, (t, uid, num_tok, max_tok) in enumerate(
            zip(y, batch.uids, batch.num_tokens, batch.max_tokens)
        ):
            num_tok += 1
            batch.num_tokens[e] = num_tok
            if t in self.stop_tokens:
                finish_reason = "stop"
                end_idx.append(e)
            elif num_tok >= max_tok:
                finish_reason = "length"
                end_idx.append(e)
            else:
                finish_reason = None
                keep_idx.append(e)
            responses.append(self.Response(uid, t, logprobs[e], finish_reason))

        # Remove any finished completions
        # if len(end_idx) > 0:
        #     print(
        #         f"Finished {len(end_idx)} completions.",
        #         end_idx,
        #         "Remaining:",
        #         len(keep_idx),
        #     )
        if len(end_idx):
            if len(keep_idx) > 0:
                batch.filter(keep_idx)
                # self.model.language_model.rope_deltas = (
                #     self.model.language_model.rope_deltas[keep_idx]
                # )
            else:
                self.active_batch = None

        self._stats.generation_tokens += len(responses)
        return responses

    def next(self):
        with mx.stream(generation_stream):
            return self._next()


def prepare_formatted_input(
    model: nn.Module,
    processor: PreTrainedTokenizer,
    prompt: list[str],
    image: Union[str, List[str]] | None = None,
    audio: Union[str, List[str]] | None = None,
    **kwargs,
):
    if kwargs.get("input_ids", None) is not None:
        input_ids = kwargs.pop("input_ids")
        pixel_values = kwargs.pop("pixel_values", None)
        mask = kwargs.pop("mask", None)
        inputs = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "attention_mask": mask,
        }
    else:
        add_special_tokens = (
            not hasattr(processor, "chat_template")
            if model.config.model_type in ["gemma3", "gemma3n"]
            else True
        )
        resize_shape = kwargs.pop("resize_shape", None)
        image_token_index = getattr(model.config, "image_token_index", None)
        inputs = prepare_inputs(
            processor,
            images=image,
            audio=audio,
            prompts=prompt,
            image_token_index=image_token_index,
            resize_shape=resize_shape,
            add_special_tokens=add_special_tokens,
        )

    return inputs


def batch_generate(
    model,
    processor,
    prompts: List[str],
    images: List[Image.Image] | None = None,
    max_tokens: Union[int, List[int]] = 2048,
    verbose: bool = False,
    **kwargs,
) -> BatchResponse:
    """
    Generate responses for the given batch of prompts.

    Args:
       model (nn.Module): The language model.
       tokenizer (PreTrainedTokenizer): The tokenizer.
       prompt (List[List[int]]): The input prompts.
       verbose (bool): If ``True``, print tokens and timing information.
          Default: ``False``.
       max_tokens (Union[int, List[int]): Maximum number of output tokens. This
          can be per prompt if a list is provided.
       kwargs: The remaining options get passed to :obj:`BatchGenerator`.
          See :obj:`BatchGenerator` for more details.
    """
    processor.tokenizer.stopping_criteria.reset(model.config.eos_token_id)
    gen = BatchGenerator(model, stop_tokens=processor.tokenizer.eos_token_ids, **kwargs)
    num_samples = len(prompts)
    fin = 0
    if verbose:
        print(f"[batch_generate] Finished processing 0/{num_samples} ...", end="\r")
    if not isinstance(prompts, list):
        prompts = [prompts]
    if images is not None:
        if not isinstance(images, list):
            images = [images]
        formatted_input = prepare_formatted_input(model, processor, prompts, images)
    else:
        formatted_input = prepare_formatted_input(model, processor, prompts)

    input_per_prompt = [
        mx.cumprod(p)[-1].item() for p in formatted_input["image_grid_thw"]
    ]
    # chunk images per prompt based on input_per_prompt
    formatted_input["pixel_values"] = [
        formatted_input["pixel_values"][
            sum(input_per_prompt[:i]) : sum(input_per_prompt[: i + 1])
        ]
        for i in range(len(input_per_prompt))
    ]

    formatted_input = [
        {k: formatted_input[k][i : i + 1] for k in formatted_input}
        for i in range(len(prompts))  # assumes only one image per prompt !!
    ]
    with wired_limit(model, [generation_stream]):
        uids = gen.insert(formatted_input, max_tokens)
        results = {uid: [] for uid in uids}
        while responses := gen.next():
            for r in responses:
                if verbose and r.finish_reason != None:
                    fin += 1
                    print(
                        f"[batch_generate] Finished processing {fin}/{num_samples} ...",
                        end="\r",
                    )
                if r.finish_reason != "stop":
                    results[r.uid].append(r.token)
    if verbose:
        print(f"[batch_generate] Finished processing {fin}/{num_samples}")

    # Return results in correct order
    texts = [processor.tokenizer.decode(results[uid]) for uid in uids]
    stats = gen.stats()
    if verbose:
        print(
            f"[batch_generate] Prompt: {stats.prompt_tokens} tokens, {stats.prompt_tps:.3f} tokens-per-sec"
        )
        print(
            f"[batch_generate] Generation: {stats.generation_tokens} tokens, "
            f"{stats.generation_tps:.3f} tokens-per-sec"
        )
        print(f"[batch_generate] Peak memory: {stats.peak_memory:.3f} GB")

    # Clear cache and free memory after batch completion
    del formatted_input, results, gen
    mx.clear_cache()

    return BatchResponse(texts, stats)
