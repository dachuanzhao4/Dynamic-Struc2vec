
from pathlib import Path
from typing import Iterable, List, Union
import logging
import numpy as np
from gensim.models import Word2Vec

logger = logging.getLogger(__name__)
__all__ = ["incremental_train"]


def incremental_train(
    prev_model_path: Union[str, Path],
    sentences: List[List[str]],
    frozen_nodes: Iterable[Union[int, str]],
    iter_inc: int = 2,
    alpha: float = 0.005,
    lambda_reg: float = 0.0,
    save_model_path: Union[str, Path, None] = None,
    save_emb_path: Union[str, Path, None] = None,
) -> Word2Vec:

    prev_model_path = Path(prev_model_path)
    model: Word2Vec = Word2Vec.load(str(prev_model_path))
    logger.info("Loaded previous model %s", prev_model_path)

    total_keys = len(model.wv.key_to_index)
    model.wv.vectors_lockf = np.ones(total_keys, dtype=model.wv.vectors_lockf.dtype)
    logger.info("Reset vectors_lockf to ones(length=%d)", total_keys)

    prev_vectors = model.wv.vectors.copy() if lambda_reg > 0 else None

    frozen_str = set(str(x) for x in frozen_nodes)
    model.build_vocab(sentences, update=True, min_count=0)
    logger.info("Vocab updated; new size=%d", len(model.wv))

    vsz = len(model.wv.key_to_index)
    lf = model.wv.vectors_lockf
    if lf.shape[0] < vsz:
        extra = np.ones(vsz - lf.shape[0], dtype=lf.dtype)
        model.wv.vectors_lockf = np.concatenate([lf, extra])
        logger.info("Extended vectors_lockf to length=%d", vsz)

    for key in frozen_str:
        if key in model.wv.key_to_index:
            idx = model.wv.key_to_index[key]
            model.wv.vectors_lockf[idx] = 0.0

    model.train(
        sentences,
        total_examples=len(sentences),
        epochs=iter_inc,
        start_alpha=alpha,
        end_alpha=alpha / 2,
    )
    if lambda_reg > 0 and prev_vectors is not None:
        for key, idx in model.wv.key_to_index.items():
            if key in frozen_str:
                continue
            if idx < prev_vectors.shape[0]:
                model.wv.vectors[idx] = (
                    (1 - lambda_reg) * model.wv.vectors[idx]
                    + lambda_reg * prev_vectors[idx]
                )

    if save_model_path is not None:
        model.save(str(save_model_path))
    if save_emb_path is not None:
        model.wv.save_word2vec_format(str(save_emb_path))

    return model
