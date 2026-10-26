import bisect

from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher


def _span_padded_size(self, num_tokens_padded: int, span: int) -> int | None:
    """Smallest capture size >= num_tokens_padded that is a multiple of ``span``,
    or None when no such bucket exists.

    ``cudagraph_capture_sizes`` is made divisible by the *config* span at config
    time, so for the static path this returns ``num_tokens_padded`` unchanged.
    A runtime span under dynamic speculative decoding has no such guarantee (for
    span=3 the 4/8/16 buckets are unusable), and padding to a non-multiple makes
    ``num_reqs = num_tokens // span`` describe a batch the graph was not captured
    for. Caching the filtered list per span keeps this O(log n) on the hot path.
    """
    cache = getattr(self, "_span_padded_sizes", None)
    if cache is None:
        cache = self._span_padded_sizes = {}
    sizes = cache.get(span)
    if sizes is None:
        capture_sizes = self.compilation_config.cudagraph_capture_sizes or []
        sizes = [s for s in capture_sizes if s % span == 0]
        cache[span] = sizes
    i = bisect.bisect_left(sizes, num_tokens_padded)
    return sizes[i] if i < len(sizes) else None


def _create_padded_batch_descriptor(
    self,
    num_tokens: int,
    uniform_decode: bool,
    has_lora: bool,
    num_active_loras: int = 0,
) -> BatchDescriptor:
    max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
    # Under dynamic speculative decoding the per-request span is the runtime
    # 1+K (stashed on the runner by the proposer hook), not the config-time
    # maximum; the stale config span mis-scales num_reqs after a K switch.
    uniform_decode_query_len = getattr(self, "_dynamic_spec_span", None) or self.uniform_decode_query_len
    num_tokens_padded = self._bs_to_padded_graph_size[num_tokens]

    # FULL mode should not be treated as uniform decode
    if (
        uniform_decode
        and self.cudagraph_mode.has_mode(CUDAGraphMode.FULL)
        and self.cudagraph_mode != CUDAGraphMode.FULL
    ):
        span_padded = _span_padded_size(self, num_tokens_padded, uniform_decode_query_len)
        if span_padded is None:
            # No usable uniform bucket for this span (batch larger than the
            # swept buckets) -> describe it as a plain mixed batch, which makes
            # dispatch fall back to a non-FULL graph instead of claiming a
            # uniform graph whose shape is wrong.
            uniform_decode = False
            num_reqs = min(num_tokens_padded, max_num_seqs)
        else:
            num_tokens_padded = span_padded
            num_reqs = min(num_tokens_padded // uniform_decode_query_len, max_num_seqs)
    else:
        uniform_decode = False
        num_reqs = min(num_tokens_padded, max_num_seqs)

    return BatchDescriptor(
        num_tokens=num_tokens_padded,
        num_reqs=num_reqs,
        uniform=uniform_decode,
        has_lora=has_lora,
        num_active_loras=num_active_loras,
    )


CudagraphDispatcher._create_padded_batch_descriptor = _create_padded_batch_descriptor
