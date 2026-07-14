
"""
Spectral Sentinel — Model Collapse Detector for GGUF Files
==========================================================

Phi(GGUF) -> Lambda in [0,1]

  Lambda = 0  -> healthy
  Lambda = 1  -> fully collapsed

Regimes:
  Lambda < 0.3        Healthy
  0.3 <= Lambda < 0.7  Warning (early-to-mid collapse)
  Lambda >= 0.7        Collapsed (terminal)

Usage:
  python spectral_sentinel.py <model.gguf>
  python spectral_sentinel.py <model.gguf> --json
  python spectral_sentinel.py <model.gguf> --verbose
  python spectral_sentinel.py <model.gguf> --dry-run
  python spectral_sentinel.py <model.gguf> --max-dim 4096

Requirements: numpy
"""

import struct, sys, json, math, os, time, argparse
import numpy as np

# ====================================================================
#  GGUF / GGML CONSTANTS
# ====================================================================

GGUF_MAGIC = 0x46554747  # "GGUF"

GGML_TYPES = {
    0:"F32", 1:"F16", 2:"Q4_0", 3:"Q4_1", 6:"Q5_0", 7:"Q5_1", 8:"Q8_0",
    10:"Q2_K", 11:"Q3_K", 12:"Q4_K", 13:"Q5_K", 14:"Q6_K", 15:"Q8_K",
    24:"I8", 25:"I16", 26:"I32", 27:"I64", 28:"F64", 30:"BF16",
}

BLOCK_ELEMS = {0:1,1:1,2:32,3:32,6:32,7:32,8:32,10:256,11:256,12:256,13:256,14:256,15:256,24:1,25:1,26:1,27:1,28:1,30:1}
BLOCK_BYTES = {0:4,1:2,2:18,3:20,6:22,7:24,8:34,10:84,11:114,12:144,13:176,14:210,15:260,24:1,25:2,26:4,27:8,28:8,30:2}
SUPPORTED   = set(BLOCK_ELEMS.keys())
QK_K        = 256

def type_name(t): return GGML_TYPES.get(t, f"?{t}")

def tensor_nbytes(t):
    bs = BLOCK_ELEMS[t['type']]; bb = BLOCK_BYTES[t['type']]
    return (t['n_elements'] // bs) * bb

# ====================================================================
#  GGUF READER
# ====================================================================

class GGUFReader:
    def __init__(self, path):
        self.path = path
        self.f = open(path, 'rb')
        self.metadata = {}
        self.tensors = []
        self.data_start = 0
        self.alignment = 32
        self._parse()

    def _parse(self):
        magic, version, n_t, n_kv = struct.unpack('<IIQQ', self.f.read(24))
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file (magic=0x{magic:08X})")
        self.version = version
        for _ in range(n_kv):
            k = self._rstr()
            self.metadata[k] = self._rval()
        self.alignment = self.metadata.get('general.alignment', 32)
        for _ in range(n_t):
            name = self._rstr()
            nd = struct.unpack('<I', self.f.read(4))[0]
            dims = list(struct.unpack(f'<{nd}Q', self.f.read(8*nd)))
            ttype = struct.unpack('<I', self.f.read(4))[0]
            off = struct.unpack('<Q', self.f.read(8))[0]
            ne = 1
            for d in dims: ne *= d
            self.tensors.append({'name':name,'dims':dims,'type':ttype,'offset':off,'n_elements':ne})
        pos = self.f.tell()
        self.data_start = (pos + self.alignment - 1) // self.alignment * self.alignment

    def _rstr(self):
        n = struct.unpack('<Q', self.f.read(8))[0]
        return self.f.read(n).decode('utf-8')

    def _rval(self):
        vt = struct.unpack('<I', self.f.read(4))[0]
        return self._rtval(vt)

    def _rtval(self, vt):
        if vt==0: return struct.unpack('<B', self.f.read(1))[0]
        if vt==1: return struct.unpack('<b', self.f.read(1))[0]
        if vt==2: return struct.unpack('<H', self.f.read(2))[0]
        if vt==3: return struct.unpack('<h', self.f.read(2))[0]
        if vt==4: return struct.unpack('<I', self.f.read(4))[0]
        if vt==5: return struct.unpack('<i', self.f.read(4))[0]
        if vt==6: return struct.unpack('<f', self.f.read(4))[0]
        if vt==7: return struct.unpack('<?', self.f.read(1))[0]
        if vt==8: return self._rstr()
        if vt==9:
            et = struct.unpack('<I', self.f.read(4))[0]
            n = struct.unpack('<Q', self.f.read(8))[0]
            return [self._rtval(et) for _ in range(n)]
        if vt==10: return struct.unpack('<Q', self.f.read(8))[0]
        if vt==11: return struct.unpack('<q', self.f.read(8))[0]
        if vt==12: return struct.unpack('<d', self.f.read(8))[0]
        raise ValueError(f"Unknown meta type {vt}")

    def read_raw(self, idx):
        t = self.tensors[idx]
        nb = tensor_nbytes(t)
        self.f.seek(self.data_start + t['offset'])
        return self.f.read(nb)

    def close(self):
        self.f.close()

# ====================================================================
#  DEQUANTIZATION
# ====================================================================

def _u4(data, n):
    qs = np.frombuffer(data, dtype=np.uint8, count=n//2)
    out = np.empty(n, dtype=np.float32)
    out[0::2] = qs & 0x0F
    out[1::2] = qs >> 4
    return out

def _u2(data, n):
    qs = np.frombuffer(data, dtype=np.uint8, count=n//4)
    out = np.empty(n, dtype=np.float32)
    out[0::4] = qs & 0x03
    out[1::4] = (qs>>2) & 0x03
    out[2::4] = (qs>>4) & 0x03
    out[3::4] = (qs>>6) & 0x03
    return out

def _u1(data, n):
    qs = np.frombuffer(data, dtype=np.uint8, count=n//8)
    out = np.empty(n, dtype=np.float32)
    for b in range(8):
        out[b::8] = (qs >> b) & 1
    return out

def _k_scales(s):
    """Unpack 8x(6-bit scale, 6-bit min) from 12 bytes. s: (nb,12) uint8."""
    d = np.zeros((s.shape[0],8), dtype=np.float32)
    m = np.zeros((s.shape[0],8), dtype=np.float32)
    d[:,:4] = s[:,:4] & 0x3F
    m[:,:4] = s[:,4:8] & 0x3F
    d[:,4:] = (s[:,8:12] & 0x0F) | ((s[:,:4] >> 6) << 4)
    m[:,4:] = (s[:,8:12] >> 4) | ((s[:,4:8] >> 6) << 4)
    return d, m

def dequantize(raw, ttype, ne):
    if ttype == 0:  return np.frombuffer(raw, dtype=np.float32).copy()
    if ttype == 1:  return np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    if ttype == 30:
        r = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        return np.frombuffer(r.tobytes(), dtype=np.float32).copy()
    if ttype == 28: return np.frombuffer(raw, dtype=np.float64).astype(np.float32)
    if ttype == 24: return np.frombuffer(raw, dtype=np.int8).astype(np.float32)
    if ttype == 25: return np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if ttype == 26: return np.frombuffer(raw, dtype=np.int32).astype(np.float32)
    if ttype == 27: return np.frombuffer(raw, dtype=np.int64).astype(np.float32)

    if ttype == 2:  # Q4_0
        nb = ne//32
        b = np.frombuffer(raw, dtype=np.dtype([('s',np.float16),('q',np.uint8,16)]), count=nb)
        sc = b['s'].astype(np.float32)
        q = _u4(b['q'].tobytes(), ne).reshape(nb,32)
        return (sc[:,None]*(q-8.0)).ravel()

    if ttype == 3:  # Q4_1
        nb = ne//32
        b = np.frombuffer(raw, dtype=np.dtype([('s',np.float16),('m',np.float16),('q',np.uint8,16)]), count=nb)
        sc = b['s'].astype(np.float32); mn = b['m'].astype(np.float32)
        q = _u4(b['q'].tobytes(), ne).reshape(nb,32)
        return (sc[:,None]*q + mn[:,None]).ravel()

    if ttype == 6:  # Q5_0
        nb = ne//32
        b = np.frombuffer(raw, dtype=np.dtype([('s',np.float16),('h',np.uint8,4),('q',np.uint8,16)]), count=nb)
        sc = b['s'].astype(np.float32)
        q4 = _u4(b['q'].tobytes(), ne).reshape(nb,32)
        h = b['h']
        q5 = q4.copy()
        for i in range(4):
            for j in range(8):
                q5[:,i*8+j] += ((h[:,i]>>j)&1).astype(np.float32) * 16.0
        return (sc[:,None]*(q5-16.0)).ravel()

    if ttype == 7:  # Q5_1
        nb = ne//32
        b = np.frombuffer(raw, dtype=np.dtype([('s',np.float16),('m',np.float16),('h',np.uint8,4),('q',np.uint8,16)]), count=nb)
        sc = b['s'].astype(np.float32); mn = b['m'].astype(np.float32)
        q4 = _u4(b['q'].tobytes(), ne).reshape(nb,32)
        h = b['h']
        q5 = q4.copy()
        for i in range(4):
            for j in range(8):
                q5[:,i*8+j] += ((h[:,i]>>j)&1).astype(np.float32) * 16.0
        return (sc[:,None]*q5 + mn[:,None]).ravel()

    if ttype == 8:  # Q8_0
        nb = ne//32
        b = np.frombuffer(raw, dtype=np.dtype([('s',np.float16),('q',np.int8,32)]), count=nb)
        sc = b['s'].astype(np.float32)
        q = b['q'].astype(np.float32)
        return (sc[:,None]*q).ravel()

    if ttype == 12:  # Q4_K
        nb = ne//QK_K
        b = np.frombuffer(raw, dtype=np.dtype([('d',np.float16),('dm',np.float16),('sc',np.uint8,12),('q',np.uint8,128)]), count=nb)
        d = b['d'].astype(np.float32); dm = b['dm'].astype(np.float32)
        sc, mn = _k_scales(b['sc'])
        scales = d[:,None]*sc; mins = dm[:,None]*mn
        q = _u4(b['q'].tobytes(), ne).reshape(nb,QK_K)
        se = np.repeat(scales,32,axis=1); me = np.repeat(mins,32,axis=1)
        return (se*q + me).ravel()

    if ttype == 13:  # Q5_K
        nb = ne//QK_K
        b = np.frombuffer(raw, dtype=np.dtype([('d',np.float16),('dm',np.float16),('sc',np.uint8,12),('h',np.uint8,32),('q',np.uint8,128)]), count=nb)
        d = b['d'].astype(np.float32); dm = b['dm'].astype(np.float32)
        sc, mn = _k_scales(b['sc'])
        scales = d[:,None]*sc; mins = dm[:,None]*mn
        q4 = _u4(b['q'].tobytes(), ne).reshape(nb,QK_K)
        h1 = _u1(b['h'].tobytes(), ne).reshape(nb,QK_K)
        q5 = q4 + h1*16.0
        se = np.repeat(scales,32,axis=1); me = np.repeat(mins,32,axis=1)
        return (se*q5 + me).ravel()

    if ttype == 14:  # Q6_K
        nb = ne//QK_K
        b = np.frombuffer(raw, dtype=np.dtype([('ql',np.uint8,128),('qh',np.uint8,64),('sc',np.int8,16),('d',np.float16)]), count=nb)
        d = b['d'].astype(np.float32)
        sc = b['sc'].astype(np.float32)
        ql = _u4(b['ql'].tobytes(), nb*QK_K).reshape(nb,QK_K)
        qh_raw = b['qh']
        qh2 = np.empty((nb,QK_K), dtype=np.float32)
        for i in range(64):
            qh2[:,i*4+0] = qh_raw[:,i] & 0x03
            qh2[:,i*4+1] = (qh_raw[:,i]>>2) & 0x03
            qh2[:,i*4+2] = (qh_raw[:,i]>>4) & 0x03
            qh2[:,i*4+3] = (qh_raw[:,i]>>6) & 0x03
        q6 = ql + qh2*16.0
        se = np.repeat(d[:,None]*sc, 16, axis=1)
        return (se*(q6-32.0)).ravel()

    if ttype == 10:  # Q2_K
        nb = ne//QK_K
        b = np.frombuffer(raw, dtype=np.dtype([('sc',np.uint8,16),('q',np.uint8,64),('d',np.float16),('dm',np.float16)]), count=nb)
        d = b['d'].astype(np.float32); dm = b['dm'].astype(np.float32)
        sr = b['sc']
        sc = (sr & 0x0F).astype(np.float32)
        mn = (sr >> 4).astype(np.float32)
        scales = d[:,None]*sc; mins = dm[:,None]*mn
        q = _u2(b['q'].tobytes(), ne).reshape(nb,QK_K)
        se = np.repeat(scales,16,axis=1); me = np.repeat(mins,16,axis=1)
        return (se*q - me).ravel()

    if ttype == 11:  # Q3_K
        nb = ne//QK_K
        b = np.frombuffer(raw, dtype=np.dtype([('hm',np.uint8,32),('q',np.uint8,64),('sc',np.uint8,16),('d',np.float16)]), count=nb)
        d = b['d'].astype(np.float32)
        sr = b['sc']
        upper = ((sr>>4)&0x0F).astype(np.float32)
        lower = (sr&0x0F).astype(np.float32)
        scales = d[:,None]*(1.0+2.0*upper)*(1.0+lower)
        q2 = _u2(b['q'].tobytes(), ne).reshape(nb,QK_K)
        h1 = _u1(b['hm'].tobytes(), ne).reshape(nb,QK_K)
        q3 = q2 + h1*4.0
        se = np.repeat(scales,16,axis=1)
        return (se*(q3-4.0)).ravel()

    if ttype == 15:  # Q8_K
        nb = ne//QK_K
        b = np.frombuffer(raw, dtype=np.dtype([('d',np.float32),('q',np.int8,QK_K)]), count=nb)
        d = b['d'].astype(np.float32)
        q = b['q'].astype(np.float32)
        return (d[:,None]*q).ravel()

    raise ValueError(f"Unsupported type {ttype} ({type_name(ttype)})")

# ====================================================================
#  MARCHENKO-PASTUR
# ====================================================================

def mp_density(x, c, s2=1.0):
    a = s2*(1-math.sqrt(c))**2
    b = s2*(1+math.sqrt(c))**2
    x = np.asarray(x, dtype=np.float64)
    r = np.zeros_like(x)
    m = (x>a)&(x<b)
    r[m] = np.sqrt((b-x[m])*(x[m]-a)) / (2*math.pi*c*s2*x[m])
    return r

_mp_cdf_cache = {}

def mp_cdf_at(x_eval, c, s2=1.0):
    key = (round(c,4), round(s2,4))
    if key not in _mp_cdf_cache:
        a = s2*(1-math.sqrt(c))**2
        b = s2*(1+math.sqrt(c))**2
        xs = np.linspace(max(a,1e-12), b, 10000)
        dx = xs[1]-xs[0]
        rho = mp_density(xs, c, s2)
        cdf = np.cumsum(rho)*dx
        cdf = np.clip(cdf, 0, 1)
        xf = np.concatenate([[0], xs, [b*1.01]])
        cf = np.concatenate([[0], cdf, [1.0]])
        _mp_cdf_cache[key] = (xf, cf)
    xf, cf = _mp_cdf_cache[key]
    return np.interp(x_eval, xf, cf)

def ks_stat_mp(eig, c, s2):
    es = np.sort(eig)
    n = len(es)
    emp = np.arange(1,n+1)/n
    theo = mp_cdf_at(es, c, s2)
    return float(max(np.max(np.abs(emp-theo)), np.max(np.abs(np.concatenate([[0],emp[:-1]])-theo))))

# ====================================================================
#  EFFECTIVE RANK BASELINE
# ====================================================================

_erf_cache = {}

def erf_mp(c):
    """Expected r_eff/min(N,M) for a healthy random matrix with aspect ratio c."""
    k = round(c, 3)
    if k in _erf_cache:
        return _erf_cache[k]
    N = 400
    M = max(int(N/k), N) if k > 0 else N
    rng = np.random.RandomState(42)
    W = rng.randn(N, M).astype(np.float64) / math.sqrt(M)
    eig = np.linalg.eigvalsh(W @ W.T)
    eig = eig[eig > 1e-12]
    sig = np.sqrt(eig)
    p = sig / sig.sum()
    H = -np.sum(p * np.log(p + 1e-30))
    frac = math.exp(H) / N
    _erf_cache[k] = frac
    return frac

# ====================================================================
#  SPECTRAL ANALYSIS
# ====================================================================

def effective_rank(eig):
    eig = eig[eig > 1e-12]
    if len(eig) == 0: return 1.0
    sig = np.sqrt(eig)
    p = sig / sig.sum()
    H = -np.sum(p * np.log(p + 1e-30))
    return float(np.exp(H))

def stable_rank(eig):
    pos = eig[eig > 0]
    if len(pos) == 0: return 0.0
    return float(pos.sum() / pos.max())

def bbp_outlier(eig, c, s2):
    b = s2 * (1 + math.sqrt(c))**2
    if b < 1e-12: return 0.0
    return float(max(0.0, (eig.max() - b) / b))

def betti_proxy(eig, c, s2):
    b = s2 * (1 + math.sqrt(c))**2
    return float(np.sum(eig > b * 1.1) / len(eig))

def analyze(W, name, ttype, max_dim):
    res = {'name':name, 'shape':list(W.shape), 'type':type_name(ttype),
           'd_mp':0,'r_eff':0,'r_eff_exp':0,'s_rank':0,'bbp':0,'betti':0,
           'lam':0, 'skipped':False, 'reason':''}

    if W.ndim != 2:
        res['skipped']=True; res['reason']=f'ndim={W.ndim}'; return res
    N, M = W.shape
    n = min(N, M)
    if n < 16:
        res['skipped']=True; res['reason']=f'small({n})'; return res
    if n > max_dim:
        res['skipped']=True; res['reason']=f'large({n}>{max_dim})'; return res

    Wf = W.astype(np.float64)
    if N <= M:
        gram = Wf @ Wf.T / M
    else:
        gram = Wf.T @ Wf / N

    try:
        eig = np.linalg.eigvalsh(gram)
    except Exception:
        res['skipped']=True; res['reason']='eigvalsh failed'; return res

    eig = np.maximum(eig, 0)
    c = n / max(N, M)
    s2 = float(np.mean(eig))
    if s2 < 1e-12:
        res['skipped']=True; res['reason']='zero variance'; return res

    eig_norm = eig / s2

    # statistics
    d_mp = ks_stat_mp(eig_norm, c, 1.0)
    r_eff = effective_rank(eig)
    erf = erf_mp(c)
    r_exp = erf * n
    sr = stable_rank(eig)
    bbp = bbp_outlier(eig, c, s2)
    bt = betti_proxy(eig, c, s2)

    # normalize to [0,1]
    lam_dmp = min(1.0, d_mp / 0.15)
    lam_rank = max(0.0, min(1.0, 1.0 - r_eff / r_exp)) if r_exp > 0 else 0.0
    lam_bbp = min(1.0, bbp / 0.5)
    lam = max(lam_dmp, lam_rank, lam_bbp)

    res.update({'d_mp':d_mp, 'r_eff':r_eff/n, 'r_eff_exp':erf,
                's_rank':sr/n, 'bbp':bbp, 'betti':bt, 'lam':lam})
    return res

# ====================================================================
#  AGGREGATION
# ====================================================================

def aggregate(results):
    valid = [r for r in results if not r['skipped']]
    if not valid:
        return {'lambda':0.0, 'regime':'UNKNOWN', 'n_analyzed':0, 'n_skipped':len(results)}

    dmps = [r['d_mp'] for r in valid]
    ranks = [r['r_eff'] for r in valid]
    exps  = [r['r_eff_exp'] for r in valid]
    sranks= [r['s_rank'] for r in valid]
    bbps  = [r['bbp'] for r in valid]
    lams  = [r['lam'] for r in valid]

    lam_dmp = min(1.0, float(np.median(dmps)) / 0.15)
    med_r = float(np.median(ranks))
    med_e = float(np.median(exps))
    lam_rank = max(0.0, min(1.0, 1.0 - med_r/med_e)) if med_e > 0 else 0.0
    lam_bbp = min(1.0, float(np.max(bbps)) / 0.5)

    lam = max(lam_dmp, lam_rank, lam_bbp)

    if lam < 0.3:   regime = 'HEALTHY'
    elif lam < 0.7: regime = 'WARNING'
    else:           regime = 'COLLAPSED'

    return {
        'lambda': round(lam, 4),
        'regime': regime,
        'lam_dmp': round(lam_dmp, 4),
        'lam_rank': round(lam_rank, 4),
        'lam_bbp': round(lam_bbp, 4),
        'd_mp_median': round(float(np.median(dmps)), 4),
        'r_eff_median': round(med_r, 4),
        'r_eff_expected': round(med_e, 4),
        's_rank_median': round(float(np.median(sranks)), 4),
        'bbp_max': round(float(np.max(bbps)), 4),
        'lam_median': round(float(np.median(lams)), 4),
        'lam_max': round(float(np.max(lams)), 4),
        'n_analyzed': len(valid),
        'n_skipped': len(results) - len(valid),
    }

# ====================================================================
#  OUTPUT
# ====================================================================

def print_dry_run(reader):
    print(f"GGUF v{reader.version} | {len(reader.tensors)} tensors | {len(reader.metadata)} metadata keys\n")
    print(f"{'Name':<45} {'Shape':<18} {'Type':<8} {'Elements':>12}  OK")
    print("-" * 95)
    for t in reader.tensors:
        shape = "x".join(str(d) for d in reversed(t['dims']))
        tn = type_name(t['type'])
        ok = "yes" if t['type'] in SUPPORTED and len(t['dims'])==2 and min(t['dims'])>=16 else "no"
        print(f"{t['name'][:45]:<45} [{shape:<16}] {tn:<8} {t['n_elements']:>12}  {ok}")

def print_summary(reader, agg, results, filepath, verbose):
    arch = reader.metadata.get('general.architecture', 'unknown')
    name = reader.metadata.get('general.name', 'unknown')
    n_params = sum(t['n_elements'] for t in reader.tensors)

    type_counts = {}
    for t in reader.tensors:
        if len(t['dims']) == 2 and t['n_elements'] > 1000:
            tn = type_name(t['type'])
            type_counts[tn] = type_counts.get(tn, 0) + 1
    primary = max(type_counts, key=type_counts.get) if type_counts else 'unknown'

    print()
    print("=" * 60)
    print("  SPECTRAL SENTINEL — Model Collapse Detector")
    print("=" * 60)
    print()
    print(f"  File:         {filepath}")
    print(f"  Name:         {name}")
    print(f"  Architecture: {arch}")
    print(f"  Quantization: {primary}")
    print(f"  Parameters:   {n_params/1e9:.2f}B ({len(reader.tensors)} tensors)")
    print(f"  Analyzed:     {agg['n_analyzed']} layers ({agg['n_skipped']} skipped)")
    print()
    print("-" * 60)
    print("  COLLAPSE SEVERITY SCORE")
    print("-" * 60)
    print()

    lam = agg['lambda']
    blen = 40
    filled = int(lam * blen)
    bar = "█" * filled + "░" * (blen - filled)
    print(f"  Lambda = {lam:.4f}  [{bar}]")
    print()
    print(f"  D_MP  (median):  {agg['d_mp_median']:.4f}  ->  lam_DMP  = {agg['lam_dmp']:.4f}")
    print(f"  r_eff (median):  {agg['r_eff_median']:.4f}  (expected: {agg['r_eff_expected']:.4f})")
    print(f"                   ->  lam_rank = {agg['lam_rank']:.4f}")
    print(f"  BBP   (max):     {agg['bbp_max']:.4f}  ->  lam_BBP  = {agg['lam_bbp']:.4f}")
    print(f"  Stable (median): {agg['s_rank_median']:.4f}")
    print()

    regime = agg['regime']
    if regime == 'HEALTHY':
        sym = "OK"; msg = "No signs of model collapse detected."
    elif regime == 'WARNING':
        sym = "!!"; msg = "Early-to-mid collapse signatures detected. Investigate further."
    else:
        sym = "XX"; msg = "Terminal collapse detected. Model weights show severe degradation."

    print(f"  Regime: [{sym}] {regime}")
    print()
    print(f"  Thresholds:")
    print(f"    Lambda < 0.3       Healthy")
    print(f"    0.3 <= Lambda < 0.7  Warning (early-to-mid collapse)")
    print(f"    Lambda >= 0.7      Collapsed (terminal)")
    print()
    print(f"  {msg}")
    print()

    if verbose:
        valid = [r for r in results if not r['skipped']]
        valid.sort(key=lambda r: r['lam'], reverse=True)
        print("-" * 100)
        print(f"  PER-LAYER STATISTICS (top 15 by Lambda)")
        print("-" * 100)
        hdr = f"  {'Tensor':<40} {'Shape':<14} {'Type':<6} {'D_MP':>6} {'r_eff':>6} {'s_rank':>7} {'BBP':>6} {'Lambda':>6}"
        print(hdr)
        print(f"  {'-'*40} {'-'*14} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*6} {'-'*6}")
        for r in valid[:15]:
            sh = f"{r['shape'][0]}x{r['shape'][1]}"
            print(f"  {r['name'][:40]:<40} {sh:<14} {r['type']:<6} "
                  f"{r['d_mp']:>6.3f} {r['r_eff']:>6.3f} {r['s_rank']:>7.3f} "
                  f"{r['bbp']:>6.3f} {r['lam']:>6.3f}")
        print()

def output_json(reader, agg, results, filepath):
    out = {
        'file': filepath,
        'name': reader.metadata.get('general.name', ''),
        'architecture': reader.metadata.get('general.architecture', 'unknown'),
        'gguf_version': reader.version,
        'n_tensors': len(reader.tensors),
        'collapse_severity': agg['lambda'],
        'regime': agg['regime'],
        'components': {
            'lambda_dmp': agg['lam_dmp'],
            'lambda_rank': agg['lam_rank'],
            'lambda_bbp': agg['lam_bbp'],
        },
        'statistics': {
            'd_mp_median': agg['d_mp_median'],
            'r_eff_median': agg['r_eff_median'],
            'r_eff_expected': agg['r_eff_expected'],
            'stable_rank_median': agg['s_rank_median'],
            'bbp_max': agg['bbp_max'],
            'lambda_layer_median': agg['lam_median'],
            'lambda_layer_max': agg['lam_max'],
        },
        'n_analyzed': agg['n_analyzed'],
        'n_skipped': agg['n_skipped'],
        'layers': [r for r in results if not r['skipped']],
    }
    print(json.dumps(out, indent=2))

# ====================================================================
#  MAIN
# ====================================================================

def main():
    p = argparse.ArgumentParser(
        description="Spectral Sentinel: Model Collapse Detector for GGUF Files")
    p.add_argument('model', help='Path to GGUF model file')
    p.add_argument('--json', action='store_true', help='Output as JSON')
    p.add_argument('--verbose', '-v', action='store_true', help='Per-layer table')
    p.add_argument('--max-dim', type=int, default=8192, help='Skip tensors where min(dim) > N')
    p.add_argument('--max-tensors', type=int, default=0, help='Limit to N tensors (0=all)')
    p.add_argument('--dry-run', action='store_true', help='List tensors only')
    args = p.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: file not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    try:
        reader = GGUFReader(args.model)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print_dry_run(reader)
        reader.close()
        return

    # Select analyzable 2D weight tensors
    candidates = []
    for i, t in enumerate(reader.tensors):
        if len(t['dims']) != 2: continue
        if t['type'] not in SUPPORTED: continue
        if min(t['dims']) < 16: continue
        candidates.append((i, t))

    if args.max_tensors > 0 and len(candidates) > args.max_tensors:
        step = len(candidates) / args.max_tensors
        candidates = [candidates[int(i*step)] for i in range(args.max_tensors)]

    if not candidates:
        print("No analyzable weight tensors found.", file=sys.stderr)
        reader.close()
        sys.exit(1)

    if not args.json:
        print(f"Analyzing {len(candidates)} weight tensors...", file=sys.stderr)

    results = []
    t0 = time.time()

    for idx, (ti, tinfo) in enumerate(candidates):
        if not args.json:
            pct = (idx+1)/len(candidates)*100
            el = time.time() - t0
            eta = el/(idx+1)*(len(candidates)-idx-1) if idx > 0 else 0
            sys.stderr.write(f"\r  [{idx+1}/{len(candidates)}] {pct:.0f}% ({el:.0f}s elapsed, ~{eta:.0f}s remaining)  ")

        try:
            raw = reader.read_raw(ti)
            W = dequantize(raw, tinfo['type'], tinfo['n_elements'])
            shape = tuple(tinfo['dims'][::-1])
            W = W.reshape(shape)
            r = analyze(W, tinfo['name'], tinfo['type'], args.max_dim)
        except Exception as e:
            r = {'name':tinfo['name'], 'shape':list(tinfo['dims'][::-1]),
                 'type':type_name(tinfo['type']), 'd_mp':0,'r_eff':0,'r_eff_exp':0,
                 's_rank':0,'bbp':0,'betti':0,'lam':0,
                 'skipped':True, 'reason':str(e)[:60]}
        results.append(r)
        if 'W' in dir(): del W

    if not args.json:
        sys.stderr.write('\n')

    agg = aggregate(results)

    if args.json:
        output_json(reader, agg, results, args.model)
    else:
        print_summary(reader, agg, results, args.model, args.verbose)

    reader.close()

if __name__ == '__main__':
    main()
