import numpy as np
from pathlib import Path
from app.storage import GalleryStore
from app.search import cosine_search


def test_cosine_ranking(tmp_path: Path):
    store = GalleryStore(tmp_path/'t.sqlite3')
    a = np.zeros(512, np.float32); a[0] = 1
    b = np.zeros(512, np.float32); b[1] = 1
    store.add('a.jpg',(0,0,1,1),a,'A')
    store.add('b.jpg',(0,0,1,1),b,'B')
    ranked = cosine_search(store, a, 2)
    assert ranked[0][0].image_id == 'a.jpg'
    assert ranked[0][1] > ranked[1][1]
