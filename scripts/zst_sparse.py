import argparse
import json
import os
import sys
import urllib.request

import zstandard

PAGE = 1 << 20


class Intervals:
    def __init__(self, path):
        self.path = path
        self.spans = json.load(open(path)) if path and os.path.exists(path) else []

    def save(self):
        json.dump(self.spans, open(self.path, "w"))

    def add(self, a, b):
        spans = sorted(self.spans + [[a, b]])
        merged = []
        for s, e in spans:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        self.spans = merged

    def missing(self, a, b):
        out, cur = [], a
        for s, e in self.spans:
            if e <= cur:
                continue
            if s >= b:
                break
            if s > cur:
                out.append([cur, min(s, b)])
            cur = max(cur, e)
            if cur >= b:
                break
        if cur < b:
            out.append([cur, b])
        return out


class SparseFile:
    def __init__(self, root, size):
        self.data_path = os.path.join(root, "sparse.bin")
        self.have = Intervals(os.path.join(root, "have.json"))
        self.misses = []
        self.size = size
        if not os.path.exists(self.data_path):
            with open(self.data_path, "wb") as f:
                f.truncate(size)
        self.f = open(self.data_path, "rb")
        self.pos = 0

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def tell(self):
        return self.pos

    def readinto(self, buf):
        n = len(buf)
        n = max(0, min(n, self.size - self.pos))
        for a, b in self.have.missing(self.pos, self.pos + n):
            self.misses.append([a, b])
        self.f.seek(self.pos)
        got = self.f.readinto(memoryview(buf)[:n])
        self.pos += got
        return got

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        buf = bytearray(n)
        got = self.readinto(buf)
        return bytes(buf[:got])


def page_ranges(spans):
    pages = Intervals(None)
    for a, b in spans:
        pages.add((a // PAGE) * PAGE, ((b + PAGE - 1) // PAGE) * PAGE)
    return pages.spans


def fetch(url, root, wanted, size, window_log=27):
    have = Intervals(os.path.join(root, "have.json"))
    wanted = [w for a, b in page_ranges(wanted) for w in have.missing(a, b)]
    if not wanted:
        return 0
    stop = max(b for _, b in wanted)
    dctx = zstandard.ZstdDecompressor(max_window_size=1 << window_log)
    req = urllib.request.Request(url, headers={"User-Agent": "zst-sparse"})
    pos, written = 0, 0
    out = open(os.path.join(root, "sparse.bin"), "r+b")
    with urllib.request.urlopen(req) as resp:
        reader = dctx.stream_reader(resp, read_size=1 << 20)
        while pos < stop:
            chunk = reader.read(8 << 20)
            if not chunk:
                break
            end = pos + len(chunk)
            for a, b in wanted:
                s, e = max(a, pos), min(b, end)
                if s < e:
                    out.seek(s)
                    out.write(chunk[s - pos : e - pos])
                    have.add(s, e)
                    written += e - s
            pos = end
            if pos // (256 << 20) != (pos - len(chunk)) // (256 << 20):
                have.save()
                print(f"decompressed {pos / 1e9:.2f} GB of {stop / 1e9:.2f} GB needed", flush=True)
    have.save()
    out.close()
    return written


def content_size(url):
    req = urllib.request.Request(url, headers={"User-Agent": "zst-sparse", "Range": "bytes=0-65535"})
    head = urllib.request.urlopen(req).read()
    return zstandard.get_frame_parameters(head).content_size


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--want", required=True)
    a = p.parse_args()
    os.makedirs(a.root, exist_ok=True)
    meta = os.path.join(a.root, "meta.json")
    if not os.path.exists(meta):
        json.dump({"url": a.url, "size": content_size(a.url)}, open(meta, "w"))
    size = json.load(open(meta))["size"]
    data = os.path.join(a.root, "sparse.bin")
    if not os.path.exists(data):
        with open(data, "wb") as f:
            f.truncate(size)
    wanted = json.load(open(a.want))
    n = fetch(a.url, a.root, wanted, size)
    print(f"size {size} wrote {n}", file=sys.stderr)
