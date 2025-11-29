import numpy as np
from src.db import save_user_embedding, load_user_embedding_db


def test_save_and_load_embedding():
    fname = 'pytest'
    lname = 'user'
    emb = np.random.rand(512).astype('float32')

    # Save embedding
    ok = save_user_embedding(fname, lname, emb)
    assert ok is True

    # Load embedding
    loaded = load_user_embedding_db(fname, lname)
    assert loaded is not None
    assert isinstance(loaded, np.ndarray)
    # Values should match
    assert np.allclose(emb, loaded)
