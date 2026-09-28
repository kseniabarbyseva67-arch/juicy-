import numpy as np
from app.embedding import VehicleEmbedder


def test_embedding_fixed_and_normalized():
    img = np.zeros((100, 120, 3), dtype=np.uint8)
    img[20:80, 30:100] = (20, 80, 180)
    e = VehicleEmbedder().embed(img, (30,20,70,60))
    assert e.shape == (512,)
    assert np.isclose(np.linalg.norm(e), 1.0, atol=1e-5)
