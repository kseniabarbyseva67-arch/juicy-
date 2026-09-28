"""Build submission.csv, candidates.csv and embeddings.npy from a CSV + images folder.

Expected CSV columns: image_id,x,y,w,h. If vehicle_id exists it is ignored for queries.
Gallery creation and query split are task-specific, so this helper exports embeddings for
all rows and demonstrates the required artifact formats.
"""
import argparse, csv
from pathlib import Path
import cv2, numpy as np
from app.embedding import VehicleEmbedder


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--images', required=True)
    p.add_argument('--out', default='artifacts')
    args=p.parse_args()
    out=Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows=list(csv.DictReader(open(args.csv, encoding='utf-8')))
    emb=VehicleEmbedder(); vecs=[]
    for r in rows:
        img=cv2.imread(str(Path(args.images)/r['image_id']))
        if img is None: raise FileNotFoundError(r['image_id'])
        bb=tuple(int(r[k]) for k in ('x','y','w','h'))
        vecs.append(emb.embed(img, bb))
    np.save(out/'embeddings.npy', np.stack(vecs))
    with open(out/'submission.csv','w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['query_id','candidates'])
        for r in rows: w.writerow([r['image_id'],''])
    with open(out/'candidates.csv','w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['query_id','candidate_id','score'])
        for r in rows: w.writerow([r['image_id'],'',''])
    print(f'Wrote {len(rows)} embeddings to {out}')

if __name__=='__main__': main()
