"""Minimal pure-Python Parquet reader (enough for HF image datasets written by pyarrow).
Supports: Thrift compact footer, data page v1/v2, PLAIN + RLE_DICTIONARY/PLAIN_DICTIONARY,
RLE/bit-packed definition levels, codecs UNCOMPRESSED and SNAPPY (pure-Python decoder)."""
import struct

# ---------------- Thrift compact protocol ----------------
class TReader:
    def __init__(self, buf, pos=0): self.b, self.p = buf, pos
    def byte(self):
        v = self.b[self.p]; self.p += 1; return v
    def varint(self):
        r = s = 0
        while True:
            x = self.byte(); r |= (x & 0x7F) << s; s += 7
            if not x & 0x80: return r
    def zz(self):
        n = self.varint(); return (n >> 1) ^ -(n & 1)
    def binary(self):
        n = self.varint(); v = self.b[self.p:self.p + n]; self.p += n; return bytes(v)
    def value(self, t):
        if t in (1, 2): return t == 1
        if t == 3: return struct.unpack('b', bytes([self.byte()]))[0]
        if t in (4, 5, 6): return self.zz()
        if t == 7:
            v = struct.unpack('<d', self.b[self.p:self.p + 8])[0]; self.p += 8; return v
        if t == 8: return self.binary()
        if t in (9, 10):
            h = self.byte(); n = h >> 4; et = h & 0x0F
            if n == 15: n = self.varint()
            return [self.value(et) for _ in range(n)]
        if t == 12: return self.struct()
        raise ValueError(f"thrift type {t}")
    def struct(self):
        out, last = {}, 0
        while True:
            h = self.byte()
            if h == 0: return out
            t = h & 0x0F; d = h >> 4
            fid = last + d if d else self.zz()
            last = fid
            out[fid] = self.value(t)

# ---------------- Snappy ----------------
def snappy_decompress(src):
    src = memoryview(src); p = 0; n = 0; s = 0
    while True:
        x = src[p]; p += 1; n |= (x & 0x7F) << s; s += 7
        if not x & 0x80: break
    out = bytearray()
    L = len(src)
    while p < L:
        tag = src[p]; p += 1; t = tag & 3
        if t == 0:
            ln = tag >> 2
            if ln >= 60:
                nb = ln - 59; ln = int.from_bytes(src[p:p + nb], 'little'); p += nb
            ln += 1
            out += src[p:p + ln]; p += ln
            continue
        if t == 1:
            ln = ((tag >> 2) & 7) + 4; off = ((tag >> 5) << 8) | src[p]; p += 1
        elif t == 2:
            ln = (tag >> 2) + 1; off = src[p] | (src[p + 1] << 8); p += 2
        else:
            ln = (tag >> 2) + 1; off = int.from_bytes(src[p:p + 4], 'little'); p += 4
        start = len(out) - off
        if off >= ln:
            out += out[start:start + ln]
        else:
            for i in range(ln): out.append(out[start + i])
    assert len(out) == n, (len(out), n)
    return bytes(out)

def decompress(codec, data, usize):
    if codec == 0: return bytes(data)
    if codec == 1: return snappy_decompress(data)
    if codec == 6:
        import zstandard; return zstandard.ZstdDecompressor().decompress(data, max_output_size=usize)
    if codec == 2:
        import gzip; return gzip.decompress(data)
    raise ValueError(f"codec {codec} not supported")

# ---------------- RLE / bit-packed hybrid ----------------
def rle_hybrid(buf, bit_width, count):
    out = []; p = 0; bw_bytes = (bit_width + 7) // 8
    r = TReader(buf)
    while len(out) < count and r.p < len(buf):
        h = r.varint()
        if h & 1:
            groups = h >> 1; nbytes = groups * bit_width
            chunk = int.from_bytes(buf[r.p:r.p + nbytes], 'little'); r.p += nbytes
            mask = (1 << bit_width) - 1
            for i in range(groups * 8):
                out.append((chunk >> (i * bit_width)) & mask)
        else:
            run = h >> 1
            v = int.from_bytes(buf[r.p:r.p + bw_bytes], 'little'); r.p += bw_bytes
            out.extend([v] * run)
    return out[:count]

# ---------------- Reading ----------------
def read_meta(path):
    with open(path, 'rb') as f:
        f.seek(-8, 2); n, magic = struct.unpack('<I4s', f.read(8))
        assert magic == b'PAR1'
        f.seek(-8 - n, 2); return TReader(f.read(n)).struct()

def schema_info(meta):
    """Return {dotted_path: (physical_type, max_def_level)} for leaf columns."""
    els = meta[2]; out = {}; i = 0
    def walk(prefix, deflev):
        nonlocal i
        el = els[i]; i += 1
        name = el[4].decode(); rep = el.get(3, 0); nch = el.get(5, 0)
        d = deflev + (1 if rep == 1 else 0)
        path = prefix + [name]
        if nch:
            for _ in range(nch): walk(path, d)
        else:
            out['.'.join(path[1:])] = (el.get(1), d)
    walk([], -0)
    # root counts as level 0 (required), fix: root element rep is absent -> d unchanged
    return out

def plain_values(buf, ptype, n):
    if ptype == 6:  # BYTE_ARRAY
        vals = []; p = 0; mv = memoryview(buf)
        for _ in range(n):
            ln = struct.unpack_from('<I', buf, p)[0]; p += 4
            vals.append(bytes(mv[p:p + ln])); p += ln
        return vals
    fmt = {1: 'i', 2: 'q', 4: 'f', 5: 'd'}[ptype]
    return list(struct.unpack_from('<%d%s' % (n, fmt), buf, 0))

def read_column(f, cmeta, ptype, max_def):
    """Yield values (None for nulls) of one column chunk."""
    codec = cmeta[4]; total = cmeta[5]
    start = cmeta.get(11) or cmeta[9]
    f.seek(start); dictionary = None; seen = 0
    while seen < total:
        hdr_buf = f.read(64 * 1024); r = TReader(hdr_buf); ph = r.struct()
        f.seek(f.tell() - len(hdr_buf) + r.p)
        ptype_page, usize, csize = ph[1], ph[2], ph[3]
        raw = f.read(csize)
        if ptype_page == 2:  # dictionary page
            data = decompress(codec, raw, usize)
            dictionary = plain_values(data, ptype, ph[7][1]); continue
        if ptype_page == 0:  # data page v1
            data = decompress(codec, raw, usize); h = ph[5]; nv = h[1]; enc = h[2]; p = 0
            if max_def > 0:
                ln = struct.unpack_from('<I', data, 0)[0]
                defs = rle_hybrid(data[4:4 + ln], max_def.bit_length(), nv); p = 4 + ln
            else:
                defs = [0] * nv
            body = data[p:]
        elif ptype_page == 3:  # data page v2
            h = ph[8]; nv = h[1]; enc = h[4]; dl = h[5]; rl = h[6]
            levels = raw[:rl + dl]
            defs = rle_hybrid(levels[rl:rl + dl], max_def.bit_length(), nv) if max_def > 0 else [0] * nv
            rest = raw[rl + dl:]
            body = decompress(codec, rest, usize - rl - dl) if h.get(7, True) else bytes(rest)
        else:
            continue
        nn = sum(1 for d in defs if d == max_def)
        if enc in (2, 8):  # PLAIN_DICTIONARY / RLE_DICTIONARY
            bw = body[0]; idx = rle_hybrid(body[1:], bw, nn) if bw else [0] * nn
            vals = [dictionary[i] for i in idx]
        elif enc == 0:
            vals = plain_values(body, ptype, nn)
        else:
            raise ValueError(f"encoding {enc}")
        it = iter(vals)
        for d in defs:
            yield next(it) if d == max_def else None
        seen += nv

def iter_rows(path, columns):
    """Yield dicts {column_path: value} row by row. columns = dotted leaf paths."""
    meta = read_meta(path); info = schema_info(meta)
    with open(path, 'rb') as f:
        for rg in meta[4]:
            cols = {}
            for cc in rg[1]:
                cm = cc[3]; name = '.'.join(x.decode() for x in cm[3])
                if name in columns:
                    ptype, md = info[name]
                    cols[name] = list(read_column(f, cm, ptype, md))
            n = rg[3]
            for i in range(n):
                yield {c: cols[c][i] for c in columns}
