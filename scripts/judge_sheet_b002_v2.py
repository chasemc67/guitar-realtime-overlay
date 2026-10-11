import cv2,numpy as np
J={'3FoeHtEf0P0': '', '8-grcbKnbps': 'BAD: misses high E side; frets 14-17 rejected', '9IDs9ol5lAE': '', 'AmyRsJqZpjo': '', 'Av9DDOjuRho': '', 'Beb0HdyBhd8': '', 'HiSaMX45AMw': '', 'I4Vt5Q4aUrw': '', 'IZJdHyfkrsg': 'looks full; GT says span off 37% - verify', 'NA9J_pQkEaE': '', 'NN224KF8WxU': 'BAD: oblique, ~1/3 of width, centred wrong', 'QYoiaH5lId8': 'BAD: high-E line tilts off board toward nut', 'RztmWh5kYuw': '', 'SGTxeiB_jBM': 'borderline: body end ok-ish (lens curve)', 'SekrUcb03Eg': '', 'XyBfN5WNo9g': '', 'Z21qSJ3HynA': '', 'aZH5YWOY2Fc': '', 'bOmRY782-GE': '', 'bVmq2C5kLoM': '', 'gQZDu0mIhwc': '', 'hAuPpjcpZSU': 'BAD: tiny patch, frets 15-24 only, ~half width', 'hqBFkJAI3dA': '', 'mP2wtBlcCSc': 'BAD: right width but shifted past bass edge', 'nbkiBItGzoU': 'BAD: total miss, grid on amp grille', 'r7bBo-b2Xh0': '', 'rtVaz9HSyZw': '', 'tWLgUEf0h40': '', 'wi9ZivDT1l0': '', 'xI3ZBCyl6Ec': '', 'xQg_N9mBoYc': '', 's7DVJX4PK3U': '(acoustic, excluded)'}
def sheet(items,out):
    W=480;tiles=[]
    for k,r in items:
        im=cv2.imread(f"frames/{k}_08_f000.jpg"); im=im[50:] # drop header
        h=int(im.shape[0]*W/im.shape[1]); im=cv2.resize(im,(W,h)); im=cv2.resize(im,(W,300)) if h!=300 else im
        cap=np.zeros((44,W,3),np.uint8)
        cv2.putText(cap,k,(6,17),0,0.5,(255,255,255),1,cv2.LINE_AA)
        if r: cv2.putText(cap,r,(6,37),0,0.45,(80,80,255) if r.startswith("BAD") else (200,200,200),1,cv2.LINE_AA)
        tiles.append(np.vstack([cap,im]))
    c=4; th=tiles[0].shape[0]
    while len(tiles)%c: tiles.append(np.zeros_like(tiles[0]))
    rows=[np.hstack(tiles[i:i+c]) for i in range(0,len(tiles),c)]
    cv2.imwrite(out,np.vstack(rows),[cv2.IMWRITE_JPEG_QUALITY,85])
good=[(k,r) for k,r in J.items() if not r.startswith("BAD") and not r.startswith("(")]
bad=[(k,r) for k,r in J.items() if r.startswith("BAD")]
sheet(good,"judged_good.jpg");sheet(bad,"judged_bad.jpg")
open("judged.csv","w").write("frame,verdict,reason\n"+"".join(f"{k},{'BAD' if r.startswith('BAD') else 'GOOD'},{r.replace('BAD: ','')}\n" for k,r in J.items()))
print(len(good),len(bad))
