#!/usr/bin/env python3
"""Video -> DDS frames. Reads video.mp4, writes DDS files into ./frames"""
import shutil, struct, subprocess, sys
from pathlib import Path

import numpy as np

# ---------------- CONFIG ----------------
VIDEO         = "video.mp4"
OUT_DIR       = "frames"
WIDTH, HEIGHT = 1280, 720     # must be multiples of 4
FPS           = None          # None = video's native fps, or e.g. 30
SIZE_LIMIT    = 16 * 1024 * 1024   # PS4 dynamic theme limit (for the warning only)
# ----------------------------------------

HERE = Path(__file__).resolve().parent


def die(msg):
    print(f"\nERROR: {msg}")
    sys.exit(1)


def ask_yes_no(prompt):
    while True:
        a = input(prompt).strip().lower()
        if a in ("y", "yes"):
            return True
        if a in ("n", "no"):
            return False
        print("Please type yes or no.")


def ask_count():
    while True:
        a = input("How many frames? ").strip()
        if a.isdigit() and int(a) > 0:
            return int(a)
        print("Please enter a whole number greater than 0.")


def encode_bc1(rgb, w, h):
    a = np.frombuffer(rgb, dtype=np.uint8).reshape(h, w, 3)
    blk = (a.reshape(h // 4, 4, w // 4, 4, 3).transpose(0, 2, 1, 3, 4)
             .reshape(-1, 16, 3).astype(np.int32))
    mx, mn = blk.max(1), blk.min(1)

    def to565(c):
        return ((c[:, 0] >> 3) << 11) | ((c[:, 1] >> 2) << 5) | (c[:, 2] >> 3)

    def from565(v):
        r = (v >> 11) & 31; g = (v >> 5) & 63; b = v & 31
        return np.stack([(r << 3) | (r >> 2), (g << 2) | (g >> 4),
                         (b << 3) | (b >> 2)], 1).astype(np.float32)

    c0, c1 = to565(mx), to565(mn)              # c0 >= c1 -> 4-colour opaque mode
    p0, p1 = from565(c0), from565(c1)
    pal = np.stack([p0, p1, (2 * p0 + p1) / 3, (p0 + 2 * p1) / 3], 1)
    d = ((blk[:, :, None, :].astype(np.float32) - pal[:, None, :, :]) ** 2).sum(3)
    idx = d.argmin(2).astype(np.uint32)
    idx[c0 == c1] = 0                          # flat block: avoid transparent index
    bits = np.zeros(len(idx), dtype=np.uint32)
    for i in range(16):
        bits |= idx[:, i] << (2 * i)
    out = np.empty(len(idx), dtype=[("c0", "<u2"), ("c1", "<u2"), ("i", "<u4")])
    out["c0"], out["c1"], out["i"] = c0, c1, bits
    return out.tobytes()


def dds_bytes(rgb, w, h):
    data = encode_bc1(rgb, w, h)
    hdr = b"DDS " + struct.pack("<7I", 124, 0x81007, h, w, (w // 4) * (h // 4) * 8, 0, 1)
    hdr += b"\x00" * 44
    hdr += struct.pack("<2I4s5I", 32, 4, b"DXT1", 0, 0, 0, 0, 0)
    hdr += struct.pack("<5I", 0x1000, 0, 0, 0, 0)
    assert len(hdr) == 128
    return hdr + data


def main():
    video = HERE / VIDEO
    if not video.exists():
        die(f"{VIDEO} not found next to the script.")
    if not shutil.which("ffmpeg"):
        die("ffmpeg not found in PATH. Install it from https://ffmpeg.org")
    if WIDTH % 4 or HEIGHT % 4:
        die("WIDTH and HEIGHT must be multiples of 4.")

    limited = ask_yes_no("Do you want limited frames from the video? (yes/no): ")
    count = ask_count() if limited else None

    per_frame = (WIDTH // 4) * (HEIGHT // 4) * 8 + 128
    if count:
        total = per_frame * count
        print(f"Estimated size: {total / 1024 / 1024:.2f} MiB "
              f"({per_frame / 1024:.0f} KiB per frame)")
        if total > SIZE_LIMIT:
            print("WARNING: that is over 16 MiB, too big for a PS4 dynamic theme.")
    else:
        print(f"Extracting ALL frames ({per_frame / 1024:.0f} KiB each). "
              f"This can be a lot of files.")

    out = HERE / OUT_DIR
    out.mkdir(exist_ok=True)
    for old in out.glob("*.dds"):
        old.unlink()

    vf = []
    if FPS:
        vf.append(f"fps={FPS}")
    vf.append(f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase:flags=lanczos")
    vf.append(f"crop={WIDTH}:{HEIGHT}")
    cmd = ["ffmpeg", "-loglevel", "error", "-ss", "0", "-i", str(video),
           "-vf", ",".join(vf)]
    if count:
        cmd += ["-frames:v", str(count)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]

    pad = max(2, len(str(count - 1))) if count else 5
    frame_size = WIDTH * HEIGHT * 3
    n = 0
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    try:
        while True:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            (out / f"{n:0{pad}d}.dds").write_bytes(dds_bytes(raw, WIDTH, HEIGHT))
            n += 1
            if n % 10 == 0 or n == count:
                print(f"  {n} frames written", end="\r", flush=True)
    finally:
        proc.stdout.close()
        proc.wait()

    print()
    if n == 0:
        die("No frames were extracted. Check that video.mp4 is a valid video.")
    if count and n < count:
        print(f"WARNING: video only had {n} frames, you asked for {count}.")
    size = sum(f.stat().st_size for f in out.glob("*.dds"))
    print(f"Done: {n} DDS frames in '{OUT_DIR}' ({size / 1024 / 1024:.2f} MiB total)")


if __name__ == "__main__":
    main()