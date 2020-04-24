import json
import os

DEFAULT_EXT = os.environ.get('DEFAULT_EXTENSION', '.h5')
DEFAULT_CHUNKS = tuple(json.loads(os.environ.get('DEFAULT_CHUNKS', '[256, 256]')))


def get_default_extension():
    return DEFAULT_EXT


def get_default_chunks(data):
    chunks = DEFAULT_CHUNKS
    shape = data.shape

    len_diff = len(shape) - len(chunks)
    if len_diff > 0:
        chunks = len_diff * (1,) + chunks
    assert len(chunks) == len(shape)

    chunks = tuple(min(ch, sh) for ch, sh in zip(chunks, shape))
    return chunks
