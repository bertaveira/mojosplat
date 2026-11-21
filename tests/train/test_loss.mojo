"""L1 + SSIM loss and its image gradient vs 3DGS's SSIM in float64 torch
autograd, and with SSIM over the windows that fit in the image (3DGUT's
fused-ssim padding="valid"); tests/data/make_golden_train.py."""

from max.gpu.host import DeviceContext
from std.math import sqrt
from std.testing import assert_true, TestSuite

from golden import read_kv
from mojosplat.core.types import dptr
from mojosplat.io.arrays import download, read_array, upload
from mojosplat.train.loss import PhotometricLoss

comptime D = "tests/data/golden/train/loss/"


def relative_error(got: List[Float32], want: List[Float32]) -> Float64:
    var num: Float64 = 0
    var den: Float64 = 0
    for i in range(len(want)):
        var d = Float64(got[i]) - Float64(want[i])
        num += d * d
        den += Float64(want[i]) * Float64(want[i])
    return sqrt(num / den)


def test_loss_and_gradient() raises:
    var meta = read_kv(D + "meta.txt")
    var H = Int(meta["H"])
    var W = Int(meta["W"])
    var ctx = DeviceContext()
    var image = upload(ctx, read_array[DType.float32](D + "image.f32"))
    var target = upload(ctx, read_array[DType.uint8](D + "target.u8"))
    var grad = ctx.enqueue_create_buffer[DType.float32](H * W * 3)
    var loss = PhotometricLoss(ctx, 0.2)
    var value = loss.backward(dptr(image), dptr(target), dptr(grad), W, H, compute_loss=True)
    var want = Float64(meta["loss"])
    print("  loss", value, "reference", want)
    assert_true(abs(value - want) < 1e-6, "loss value")

    var rel = relative_error(download(ctx, grad, H * W * 3), read_array[DType.float32](D + "grad.f32"))
    print("  gradient relative error", rel)
    assert_true(rel < 1e-5, "gradient")

    var psnr = loss.psnr(dptr(image), dptr(target), W, H)
    print("  psnr", psnr, "reference", meta["psnr"])
    assert_true(abs(psnr - Float64(meta["psnr"])) < 1e-4, "psnr")
    _ = image^
    _ = target^


def test_valid_ssim() raises:
    var meta = read_kv(D + "meta.txt")
    var H = Int(meta["H"])
    var W = Int(meta["W"])
    var ctx = DeviceContext()
    var image = upload(ctx, read_array[DType.float32](D + "image.f32"))
    var target = upload(ctx, read_array[DType.uint8](D + "target.u8"))
    var grad = ctx.enqueue_create_buffer[DType.float32](H * W * 3)
    var loss = PhotometricLoss(ctx, 0.2)
    loss.valid_ssim = True
    var value = loss.backward(dptr(image), dptr(target), dptr(grad), W, H, compute_loss=True)
    var want = Float64(meta["loss_valid"])
    print("  loss", value, "reference", want)
    assert_true(abs(value - want) < 1e-6, "loss value")
    var rel = relative_error(download(ctx, grad, H * W * 3), read_array[DType.float32](D + "grad_valid.f32"))
    print("  gradient relative error", rel)
    assert_true(rel < 1e-5, "gradient")
    _ = image^
    _ = target^


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
