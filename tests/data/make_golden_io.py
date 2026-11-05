#!/usr/bin/env python3
"""Fixtures for the I/O and dataset tests (tests/io, tests/train).

Writes into tests/data/golden/io/:

    colmap_tiny/{cameras,images,points3D}.bin
        a small COLMAP binary model written with `struct` from the format
        spec; the expected values are the constants below (mirrored in
        tests/io/test_colmap.mojo)
    tiny.jpg, tiny_rgb.u8
        a 37x23 JPEG (PIL) and its PIL-decoded [H, W, 3] pixels
    tiny.ply
        a 3DGS PLY (numpy) with N=5, K=3; property p of Gaussian g has the
        value 100 * g + p (property order x y z nx ny nz f_dc_* f_rest_*
        opacity scale_* rot_*)
    garden.txt  (only if data/mipnerf360/garden exists)
        key=value summary of the real scene computed independently here
        (counts, a pose, point sums, split sizes, scene extent, decoded
        image statistics) for tests/train/test_dataset.mojo

Usage (repo root):  pixi run -e torch python tests/data/make_golden_io.py
"""

import struct
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent / "golden" / "io"
GARDEN = ROOT / "data" / "mipnerf360" / "garden"

NUM_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12, 11: 16}


def write_colmap_tiny(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    cams = [
        (1, 1, 640, 480, [500.5, 501.25, 320.0, 240.5]),  # PINHOLE
        (7, 4, 100, 50, [1.0, 2.0, 3.0, 4.0, 0.1, 0.2, 0.3, 0.4]),  # OPENCV
    ]
    with open(d / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", len(cams)))
        for cid, model, w, h, params in cams:
            f.write(struct.pack("<iiQQ", cid, model, w, h))
            f.write(struct.pack(f"<{len(params)}d", *params))
    images = [
        (3, (0.5, 0.5, 0.5, 0.5), (1.0, 2.0, 3.0), 1, "c.png", 2),
        (1, (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 4.0), 1, "a.png", 0),
        (2, (0.0, 1.0, 0.0, 0.0), (-1.0, 0.5, 2.0), 7, "b.png", 1),
    ]
    with open(d / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for iid, q, t, cid, name, n2d in images:
            f.write(struct.pack("<i4d3di", iid, *q, *t, cid))
            f.write(name.encode() + b"\0")
            f.write(struct.pack("<Q", n2d))
            for k in range(n2d):
                f.write(struct.pack("<ddq", 10.5 + k, 20.25 + k, 100 + k))
    points = [  # id, xyz, rgb, error, track length
        (11, (0.5, -1.0, 2.0), (255, 0, 10), 0.25, 1),
        (12, (1.5, 2.5, -3.5), (1, 2, 3), 0.5, 0),
        (13, (-0.25, 0.0, 9.0), (100, 150, 200), 1.0, 3),
        (14, (7.0, 8.0, 9.5), (0, 0, 0), 2.0, 2),
    ]
    with open(d / "points3D.bin", "wb") as f:
        f.write(struct.pack("<Q", len(points)))
        for pid, xyz, rgb, err, tl in points:
            f.write(struct.pack("<Q3d3Bd", pid, *xyz, *rgb, err))
            f.write(struct.pack("<Q", tl))
            for k in range(tl):
                f.write(struct.pack("<ii", 1 + k, 5 * k))


def write_jpeg(d: Path):
    h, w = 23, 37
    y, x = np.mgrid[0:h, 0:w]
    img = np.stack([x * 255 // (w - 1), y * 255 // (h - 1), (x * y) % 256], -1).astype(np.uint8)
    Image.fromarray(img).save(d / "tiny.jpg", quality=92)
    decoded = np.asarray(Image.open(d / "tiny.jpg").convert("RGB"))
    assert decoded.shape == (h, w, 3)
    decoded.astype(np.uint8).tofile(d / "tiny_rgb.u8")


def write_ply(d: Path):
    n, k = 5, 3
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(3 * k)]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    arr = np.zeros(n, dtype=[(nm, "<f4") for nm in names])
    for g in range(n):
        for p, nm in enumerate(names):
            arr[nm][g] = 100 * g + p
    header = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n
    header += "".join(f"property float {nm}\n" for nm in names) + "end_header\n"
    with open(d / "tiny.ply", "wb") as f:
        f.write(header.encode())
        f.write(arr.tobytes())


def quat_to_rot(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def garden_summary(d: Path):
    sp = GARDEN / "sparse" / "0"
    b = (sp / "cameras.bin").read_bytes()
    ncam = struct.unpack_from("<Q", b, 0)[0]
    cid, model, cw, ch = struct.unpack_from("<iiQQ", b, 8)
    cparams = struct.unpack_from(f"<{NUM_PARAMS[model]}d", b, 32)

    b = (sp / "images.bin").read_bytes()
    nimg = struct.unpack_from("<Q", b, 0)[0]
    off, images = 8, []
    for _ in range(nimg):
        iid, qw, qx, qy, qz, tx, ty, tz, icid = struct.unpack_from("<i4d3di", b, off)
        off += 64
        end = b.index(b"\0", off)
        name = b[off:end].decode()
        off = end + 1
        n2d = struct.unpack_from("<Q", b, off)[0]
        off += 8 + 24 * n2d
        images.append((name, np.array([qw, qx, qy, qz]), np.array([tx, ty, tz])))
    assert off == len(b)
    images.sort(key=lambda t: t[0])

    b = (sp / "points3D.bin").read_bytes()
    npts = struct.unpack_from("<Q", b, 0)[0]
    off, xyz_sum, rgb_sum = 8, np.zeros(3), np.zeros(3)
    for _ in range(npts):
        _, x, y, z, r, g, bb, _ = struct.unpack_from("<Q3d3Bd", b, off)
        off += 43
        tl = struct.unpack_from("<Q", b, off)[0]
        off += 8 + 8 * tl
        xyz_sum += np.float32([x, y, z]).astype(np.float64)
        rgb_sum += (r, g, bb)
    assert off == len(b)

    test_idx = list(range(0, nimg, 8))
    centers = []
    for i, (_, q, t) in enumerate(images):
        if i in test_idx:
            continue
        R = quat_to_rot(q)
        centers.append(-R.T @ t)
    centers = np.array(centers)
    extent = 1.1 * np.linalg.norm(centers - centers.mean(0), axis=1).max()

    name0, q0, t0 = images[0]
    R0 = quat_to_rot(q0)
    img0 = np.asarray(Image.open(GARDEN / "images_4" / name0).convert("RGB"))
    h, w = img0.shape[:2]
    sx, sy = w / cw, h / ch
    lines = {
        "n_cameras": ncam, "camera_model": model, "camera_width": cw, "camera_height": ch,
        "n_images": nimg, "n_points": npts, "n_test": len(test_idx), "n_train": nimg - len(test_idx),
        "first_name": images[0][0], "last_name": images[-1][0],
        "image_width": w, "image_height": h,
        "fx": cparams[0] * sx, "fy": cparams[1] * sy, "cx": cparams[2] * sx, "cy": cparams[3] * sy,
        "view0": " ".join(f"{v:.9g}" for v in np.concatenate([R0, t0[:, None]], 1).reshape(-1)),
        "xyz_sum": " ".join(f"{v:.9g}" for v in xyz_sum),
        "rgb_sum": " ".join(f"{int(v)}" for v in rgb_sum),
        "scene_extent": f"{extent:.9g}",
        "img0_sum": int(img0.astype(np.int64).sum()),
        "img0_px": " ".join(str(int(v)) for v in img0[h // 2, w // 3]),
    }
    (d / "garden.txt").write_text("".join(f"{k}={v}\n" for k, v in lines.items()))
    print("garden:", lines["n_images"], "images,", lines["n_points"], "points, extent", lines["scene_extent"])


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    write_colmap_tiny(OUT / "colmap_tiny")
    write_jpeg(OUT)
    write_ply(OUT)
    if GARDEN.exists():
        garden_summary(OUT)
    else:
        print(f"{GARDEN} not found: skipping garden.txt")
    print("written to", OUT)
