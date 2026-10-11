import cv2,numpy as np
J={
"3FoeHtEf0P0":"BAD: covers ~half, misses G+B+high E",
"8-grcbKnbps":"BAD: thin band, only top ~2 strings",
"9IDs9ol5lAE":"",
"AmyRsJqZpjo":"",
"Av9DDOjuRho":"BAD: covers ~half, misses low E+A+D",
"Beb0HdyBhd8":"BAD: sliver ~1/5, middle strings only",
"HiSaMX45AMw":"BAD: covers ~half (middle), short both sides",
"I4Vt5Q4aUrw":"BAD: covers ~half, misses treble side",
"IZJdHyfkrsg":"BAD: ~2/3, misses B+high E",
"NA9J_pQkEaE":"",
"NN224KF8WxU":"BAD: ~1/3 of width, frets 8-23 only",
"QYoiaH5lId8":"BAD: ~1/3, middle strings only",
"RztmWh5kYuw":"BAD: sliver at top, misses 4-5 strings",
"SGTxeiB_jBM":"BAD: grid drifts + narrows toward body end",
"SekrUcb03Eg":"",
"XyBfN5WNo9g":"",
"Z21qSJ3HynA":"",
"aZH5YWOY2Fc":"BAD: covers ~half (middle), misses both E",
"bOmRY782-GE":"",
"bVmq2C5kLoM":"",
"gQZDu0mIhwc":"",
"hAuPpjcpZSU":"BAD: only frets 18-22 snapped, rest extrapolated/off",
"hqBFkJAI3dA":"",
"mP2wtBlcCSc":"BAD: sliver ~1/6 near low E",
"nbkiBItGzoU":"BAD: total miss, grid on watermark text",
"r7bBo-b2Xh0":"",
"rtVaz9HSyZw":"",
"s7DVJX4PK3U":"(acoustic, excluded)",
"tWLgUEf0h40":"BAD: ~60%, misses low E side",
"wi9ZivDT1l0":"BAD: covers ~half, misses low E+A+D",
"xI3ZBCyl6Ec":"",
"xQg_N9mBoYc":"BAD: sliver ~1/5 at top, misses 4-5 strings",
}
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
good=[(k,r) for k,r in J.items() if not r.startswith("BAD")]
bad=[(k,r) for k,r in J.items() if r.startswith("BAD")]
sheet(good,"judged_good.jpg");sheet(bad,"judged_bad.jpg")
open("judged.csv","w").write("frame,verdict,reason\n"+"".join(f"{k},{'BAD' if r.startswith('BAD') else 'GOOD'},{r.replace('BAD: ','')}\n" for k,r in J.items()))
print(len(good),len(bad))
