"""The store's per-token path, compiled (Nov 18; James: "absolutely"). The same algorithms as core.py (Matcher, IntIndex,
Pool) and core3.py (PagedPairSlots), statement for statement, so the store is bit-identical (tests/store_jit.py against
the Python path). numpy's PCG64 and its bounded draws (Generator.integers(0, n, size=8)) are reproduced exactly, so
evictions pick the same candidates and a checkpoint continues the same way on either path.

Numba is optional: without it core.py keeps its Python path (the app's bundled Python may not have it).
"""
import math

import numpy as np

import os

try:
    if os.environ.get("SOMA_NOJIT"):                     # the Python path, for checking this one against it
        raise ImportError
    if ".app/Contents/" in __file__:                    # inside soma.app: compiled code is cached outside the signed
        os.environ.setdefault("NUMBA_CACHE_DIR",        # bundle, never written into it (1.4.0)
                              os.path.expanduser("~/Library/Caches/soma/numba"))
    from numba import njit
    HAVE = True
except Exception:                                       # pragma: no cover
    HAVE = False

    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f

U = np.uint64
M32 = U(0xFFFFFFFF)
MUL_HI = U(2549297995355413924)
MUL_LO = U(4865540595714422341)
HMUL = U(0x9E3779B97F4A7C15)
LB = 6


# ---- numpy's PCG64 (XSL-RR 128/64) and Generator.integers(0, n) for n <= 2^32 (Lemire, unbuffered uint32) ----
def pcg_pack(state: dict) -> np.ndarray:
    s, inc = int(state["state"]["state"]), int(state["state"]["inc"])
    m = (1 << 64) - 1
    return np.array([s >> 64, s & m, inc >> 64, inc & m, int(state["has_uint32"]), int(state["uinteger"])], np.uint64)


def pcg_unpack(a: np.ndarray) -> dict:
    return {"bit_generator": "PCG64", "state": {"state": (int(a[0]) << 64) | int(a[1]), "inc": (int(a[2]) << 64) | int(a[3])},
            "has_uint32": int(a[4]), "uinteger": int(a[5])}


@njit(cache=True, inline="always")
def _mul64(a, b):
    al, ah = a & M32, a >> U(32)
    bl, bh = b & M32, b >> U(32)
    p0, p1, p2, p3 = al * bl, al * bh, ah * bl, ah * bh
    mid = (p0 >> U(32)) + (p1 & M32) + (p2 & M32)
    lo = (p0 & M32) | (mid << U(32))
    hi = p3 + (p1 >> U(32)) + (p2 >> U(32)) + (mid >> U(32))
    return hi, lo


@njit(cache=True)
def pcg_next64(s):
    hi, lo = _mul64(s[1], MUL_LO)
    hi = hi + s[0] * MUL_LO + s[1] * MUL_HI
    lo2 = lo + s[3]
    c = U(1) if lo2 < lo else U(0)
    hi = hi + s[2] + c
    s[0] = hi
    s[1] = lo2
    x = hi ^ lo2
    rot = hi >> U(58)
    return (x >> rot) | (x << ((U(64) - rot) & U(63)))


@njit(cache=True)
def pcg_next32(s):
    if s[4] != U(0):
        s[4] = U(0)
        return s[5]
    n = pcg_next64(s)
    s[4] = U(1)
    s[5] = n >> U(32)
    return n & M32


@njit(cache=True)
def pcg_below(s, n):
    """Generator.integers(0, n) for 1 < n < 2^32: one draw of numpy's bounded uint32 (Lemire's method)"""
    rng = U(n - 1)
    excl = rng + U(1)
    m = pcg_next32(s) * excl
    left = m & M32
    if left < excl:
        thr = (M32 - rng) % excl
        while left < thr:
            m = pcg_next32(s) * excl
            left = m & M32
    return np.int64(m >> U(32))


# ---- IntIndex (core.py): open addressing, linear probing, backward-shift deletion ----
@njit(cache=True, inline="always")
def _home(k, bits):
    return np.int64((U(k) * HMUL) >> U(64 - bits))


@njit(cache=True)
def idx_get(K, V, bits, mask, k):
    i = _home(k, bits)
    while True:
        x = K[i]
        if x == k:
            return np.int64(V[i])
        if x < 0:
            return np.int64(-1)
        i = (i + 1) & mask


@njit(cache=True)
def idx_set(K, V, bits, mask, k, v):
    i = _home(k, bits)
    while K[i] >= 0 and K[i] != k:
        i = (i + 1) & mask
    K[i] = k
    V[i] = v


@njit(cache=True)
def idx_delete(K, V, bits, mask, k):
    i = _home(k, bits)
    while K[i] != k:
        if K[i] < 0:
            return
        i = (i + 1) & mask
    j = i
    while True:
        j = (j + 1) & mask
        x = K[j]
        if x < 0:
            break
        h = _home(x, bits)
        if (j > i and (h <= i or h > j)) or (j < i and (h <= i and h > j)):
            K[i] = x
            V[i] = V[j]
            i = j
    K[i] = -1


# ---- Pool.touch / lookup over a tick ----
@njit(cache=True)
def pool_tick(K, V, bits, mask, pkeys, score, last, meta, P, lg, rng, hist, need, offs, t0h, clock, C, learn,
              ids, rec_pos, rec_node):
    """one order's nodes for a tick. hist: the `need` tokens before the tick, then the tick's tokens; offs: the key's
    offsets back (key order). meta: [nxt, recycles]. -> number of evictions (rec_pos/rec_node filled in order)"""
    w = offs.shape[0]
    mx = 0
    for a in range(w):
        if offs[a] > mx:
            mx = offs[a]
    ne = 0
    for i in range(C):
        if t0h + i < mx:
            continue
        kk = np.int64(0)
        for a in range(w):
            kk |= np.int64(hist[need + i - offs[a]]) << np.int64(16 * a)
        if not learn:
            ids[i] = idx_get(K, V, bits, mask, kk)
            continue
        t = clock + i
        n = idx_get(K, V, bits, mask, kk)
        if n < 0:
            if meta[0] < P:
                n = meta[0]
                meta[0] += 1
            else:
                # 1.4.0: nodes touched in this tick (last >= clock) are never evicted in it (Pool.touch)
                best, bs = np.int64(-1), 0.0
                for r in range(8):
                    for q in range(8):
                        c = pcg_below(rng, P)
                        if last[c] >= clock:
                            continue
                        v = score[c] * math.exp(lg * (t - last[c]))
                        if best < 0 or v < bs:
                            best, bs = c, v
                    if best >= 0:
                        break
                if best < 0:
                    ids[i] = -1
                    continue
                n = best
                ok = np.int64(0)
                for a in range(w):
                    ok |= np.int64(pkeys[n, a]) << np.int64(16 * a)
                idx_delete(K, V, bits, mask, ok)
                rec_pos[ne] = i
                rec_node[ne] = n
                ne += 1
                meta[1] += 1
            idx_set(K, V, bits, mask, kk, n)
            for a in range(w):
                pkeys[n, a] = hist[need + i - offs[a]]
            score[n] = 0.0
            last[n] = t
        score[n] = score[n] * math.exp(lg * (t - last[n])) + 1.0
        last[n] = t
        ids[i] = n
    return ne


# ---- PagedPairSlots over a tick ----
@njit(cache=True, inline="always")
def _put(ptab, ptok, ppid, R, n, j, y, p):
    g = ptab[n, j // R]
    if g < 0:
        g += ptok.shape[0]                               # numpy's -1: the dummy page (as the Python path)
    ptok[g, j % R] = y
    ppid[g, j % R] = p


@njit(cache=True)
def _alloc(ptab, ptok, ppid, R, pcnt, plast, pnode, pslot, free, mt, NP, lg, rng, ch):
    if mt[0] > 0:
        mt[0] -= 1
        return np.int64(free[mt[0]])
    cand = np.empty(8, np.int64)
    for _ in range(4):
        for q in range(8):
            cand[q] = pcg_below(rng, NP)
        best, bs = np.int64(-1), 0.0
        for q in range(8):
            c = cand[q]
            if plast[c] != ch:
                v = pcnt[c] * np.exp(lg * (ch - plast[c]))
                if best < 0 or v < bs:
                    best, bs = c, v
        if best >= 0:
            n2, s2 = pnode[best], pslot[best]
            if n2 >= 0:
                _put(ptab, ptok, ppid, R, n2, s2, -1, -1)
            mt[4] += 1
            return best
    return np.int64(-1)


@njit(cache=True)
def _reset(ptab, ptok, ppid, pgfree, pnode, free, mt, NG, R, n):
    for a in range(NG):
        g = ptab[n, a]
        if g >= 0:
            for b in range(R):
                p = ppid[g, b]
                if p >= 0:
                    free[mt[0]] = p
                    mt[0] += 1
                    pnode[p] = -1
    cnt = 0
    for a in range(NG):
        if ptab[n, a] >= 0:
            cnt += 1
    k = 0
    for a in range(NG - 1, -1, -1):                      # g[::-1]: the node's pages in reverse
        g = ptab[n, a]
        if g >= 0:
            for b in range(R):
                ptok[g, b] = -1
                ppid[g, b] = -1
            pgfree[mt[6] + k] = g
            k += 1
    mt[6] += cnt
    for a in range(NG):
        ptab[n, a] = -1


@njit(cache=True)
def assign_tick(ptab, ptok, ppid, pgfree, pcnt, plast, pnode, pslot, free, mt, NG, R, K, NP, lg, rng,
                ids, ys, rec_pos, rec_node, nrec, ch, sk, pk, new):
    """core3._assign for one order, paged slots. mt: [nfree, overflows, trash, assignments, pair_evictions,
    page_fallbacks, npgfree]. -> number of new pairs (in `new`)"""
    C = ys.shape[0]
    e = 0
    nn = 0
    for i in range(C):
        while e < nrec and rec_pos[e] == i:
            _reset(ptab, ptok, ppid, pgfree, pnode, free, mt, NG, R, rec_node[e])
            e += 1
        n = ids[i]
        if n < 0:
            continue
        y = ys[i]
        na = 0
        for a in range(NG):
            if ptab[n, a] >= 0:
                na += 1
        hit, emp = -1, -1
        for a in range(na):
            g = ptab[n, a]
            if g < 0:
                g += ptok.shape[0]
            for b in range(R):
                v = ptok[g, b]
                if hit < 0 and v == y:
                    hit = a * R + b
                if emp < 0 and v < 0:
                    emp = a * R + b
            if hit >= 0:
                break
        if hit >= 0:
            g = ptab[n, hit // R]
            if g < 0:
                g += ptok.shape[0]
            p = np.int64(ppid[g, hit % R])
            if pcnt[p] < 2147483647:
                pcnt[p] += 1
            plast[p] = ch
            sk[i] = hit
            pk[i] = p
            continue
        if emp >= 0:
            j = emp
        elif na < NG and mt[6] > 0:
            j = na * R
        else:
            j = -1
            if na < NG:
                mt[5] += 1
        if j >= 0:
            p = _alloc(ptab, ptok, ppid, R, pcnt, plast, pnode, pslot, free, mt, NP, lg, rng, ch)
            if p < 0:
                mt[2] += 1
                continue                                   # (K, NP_k): the defaults in sk, pk
            q = j // R
            if ptab[n, q] < 0:
                mt[6] -= 1
                ptab[n, q] = pgfree[mt[6]]
        else:
            mt[1] += 1
            best, bc = -1, 0
            for a in range(na):
                g = ptab[n, a]
                if g < 0:
                    g += ptok.shape[0]
                for b in range(R):
                    pr = ppid[g, b]
                    if pr >= 0 and plast[pr] != ch:
                        if best < 0 or pcnt[pr] < bc:
                            best, bc = a * R + b, pcnt[pr]
            if best < 0:
                mt[2] += 1
                continue
            j = best
            g = ptab[n, j // R]
            if g < 0:
                g += ptok.shape[0]
            p = np.int64(ppid[g, j % R])
        _put(ptab, ptok, ppid, R, n, j, y, p)
        pnode[p] = n
        pslot[p] = j
        pcnt[p] = 1
        plast[p] = ch
        mt[3] += 1
        sk[i] = j
        pk[i] = p
        new[nn] = p
        nn += 1
    return nn


# ---- the Matcher: ring, match model, and the `last` map as a hash table (stale entries never match: t - q < ring) ----
# ms: [t, mp, ml, ready, cur_tok, cur_bucket, used]; lk/lv: keys (-1 empty) and last times
@njit(cache=True)
def _lt_get(lk, lv, k):
    mask = lk.shape[0] - 1
    i = _home(k, 63 - _clz(lk.shape[0]))
    while True:
        x = lk[i]
        if x == k:
            return lv[i]
        if x < 0:
            return np.int64(-1)
        i = (i + 1) & mask


@njit(cache=True, inline="always")
def _clz(n):
    b = 0
    while (1 << (b + 1)) <= n:
        b += 1
    return 63 - b


@njit(cache=True)
def _lt_set(lk, lv, ms, k, v):
    mask = lk.shape[0] - 1
    i = _home(k, 63 - _clz(lk.shape[0]))
    while lk[i] >= 0 and lk[i] != k:
        i = (i + 1) & mask
    if lk[i] < 0:
        ms[6] += 1
    lk[i] = k
    lv[i] = v


@njit(cache=True)
def lt_rebuild(lk, lv, ms, t, r, off):
    """keep the entries still within a ring of t (after shifting by off); the dict's prune and rebase"""
    n = 0
    tk = np.empty(lk.shape[0], np.int64)
    tv = np.empty(lk.shape[0], np.int64)
    for i in range(lk.shape[0]):
        if lk[i] >= 0:
            v = lv[i] - off
            if v >= 0 and t - v < r:
                tk[n] = lk[i]
                tv[n] = v
                n += 1
            lk[i] = -1
    ms[6] = 0
    for i in range(n):
        _lt_set(lk, lv, ms, tk[i], tv[i])


@njit(cache=True)
def m_peek(R, ms, lk, lv, r, mn):
    if ms[3]:
        return
    t = ms[0]
    if ms[1] >= 0 and t - ms[1] < r:
        ms[2] += 1
    else:
        ms[1], ms[2] = -1, 0
    if ms[1] < 0 and t >= mn:
        k = np.int64(0)
        for a in range(mn):
            k |= R[(t - mn + a) % r] << np.int64(16 * a)
        q = _lt_get(lk, lv, k)
        if q >= 0 and t - q < r:
            ms[1], ms[2] = q, mn
            while (ms[2] < 32 and q - ms[2] - 1 >= 0 and t - (q - ms[2] - 1) < r
                   and R[(q - ms[2] - 1) % r] == R[((t - ms[2] - 1) % r + r) % r]):
                ms[2] += 1
    if ms[1] >= 0:
        x = max(1, ms[2] - mn + 1)
        lg2 = 0
        while (1 << (lg2 + 1)) <= x:
            lg2 += 1
        ms[4], ms[5] = R[ms[1] % r], min(LB - 1, 1 + lg2)
    else:
        ms[4], ms[5] = -1, 0
    ms[3] = 1


@njit(cache=True)
def m_push(R, ms, lk, lv, r, mn, y, rebase_at):
    m_peek(R, ms, lk, lv, r, mn)
    t = ms[0]
    R[t % r] = y
    if ms[1] >= 0:
        ms[1] = ms[1] + 1 if R[ms[1] % r] == y else -1
    if t >= mn:
        k = np.int64(0)
        for a in range(mn):
            k |= R[(t - mn + a) % r] << np.int64(16 * a)
        _lt_set(lk, lv, ms, k, t)
        if ms[6] * 2 > lk.shape[0]:
            lt_rebuild(lk, lv, ms, t, r, 0)
    ms[0] = t + 1
    ms[3] = 0
    if ms[0] >= rebase_at:
        off = (ms[0] // r - 1) * r
        if off > 0:
            ms[0] -= off
            if ms[1] >= 0:
                ms[1] -= off
            lt_rebuild(lk, lv, ms, ms[0], np.int64(1) << np.int64(62), off)


@njit(cache=True)
def m_tick(R, ms, lk, lv, r, mn, ys, W, rebase_at, mp, mb, win):
    """contexts' matcher part for a tick: window and match prediction before each token, then push it"""
    for i in range(ys.shape[0]):
        t = ms[0]
        for j in range(W):
            s = t - W + j
            win[i, j] = R[s % r] if s >= 0 else 0
        m_peek(R, ms, lk, lv, r, mn)
        mp[i], mb[i] = ms[4], ms[5]
        m_push(R, ms, lk, lv, r, mn, ys[i], rebase_at)


@njit(cache=True)
def lt_load(lk, lv, ms, keys, vals, t, r):
    for i in range(keys.shape[0]):
        _lt_set(lk, lv, ms, keys[i], vals[i])
    if ms[6] * 2 > lk.shape[0]:
        lt_rebuild(lk, lv, ms, t, r, 0)


@njit(cache=True)
def node_pairs(ptab, ppid, R, nodes):
    """core3._node_pairs for paged slots: the nodes' pair ids in row-major (node, slot) order, and each one's row"""
    NG = ptab.shape[1]
    n = 0
    for i in range(nodes.shape[0]):
        for a in range(NG):
            g = ptab[nodes[i], a]
            if g >= 0:
                for b in range(R):
                    if ppid[g, b] >= 0:
                        n += 1
    pp = np.empty(n, np.int64)
    ow = np.empty(n, np.int64)
    n = 0
    for i in range(nodes.shape[0]):
        for a in range(NG):
            g = ptab[nodes[i], a]
            if g >= 0:
                for b in range(R):
                    p = ppid[g, b]
                    if p >= 0:
                        pp[n] = p
                        ow[n] = i
                        n += 1
    return pp, ow


@njit(cache=True)
def maps_tick(ptab, ptok, ppid, R, K, NPk, nodes, M, Pm):
    """core3._maps2 for paged slots, int32: M (C, K+1) the slot tokens (-1 empty; column K the trash slot, empty),
    Pm (C, K+1) the slot pair ids (NPk where empty). Rows of nodes < 0 stay empty."""
    C = nodes.shape[0]
    NG = ptab.shape[1]
    for i in range(C):
        for j in range(K + 1):
            M[i, j] = -1
            Pm[i, j] = NPk
        n = nodes[i]
        if n < 0:
            continue
        for a in range(NG):
            g = ptab[n, a]
            if g < 0:
                continue
            for b in range(R):
                M[i, a * R + b] = ptok[g, b]
                p = ppid[g, b]
                if p >= 0:
                    Pm[i, a * R + b] = p
