"""Shared math constants for real spherical harmonics (degree <= 3)."""

comptime SH_C0: Float32 = 0.28209479177387814
comptime SH_C1: Float32 = 0.48860251190291987
comptime SH_C2A: Float32 = 1.0925484305920792
comptime SH_C2B: Float32 = 0.94617469575755997
comptime SH_C2C: Float32 = 0.31539156525251999
comptime SH_C2D: Float32 = 0.54627421529603959
comptime SH_C3A: Float32 = 0.59004358992664352
comptime SH_C3B: Float32 = 1.7701307697799304
comptime SH_C3C: Float32 = 2.8906114426405538
comptime SH_C3D: Float32 = 0.45704579946446572
comptime SH_C3E: Float32 = 2.2852289973223288
comptime SH_C3F: Float32 = 1.865881662950577
comptime SH_C3G: Float32 = 1.1195289977703462
comptime SH_C3H: Float32 = 1.4453057213202769
comptime SH_C2E: Float32 = 1.8923493915151202
comptime SH_C3I: Float32 = 3.5402615395598609
comptime SH_C3J: Float32 = 4.5704579946446566
comptime SH_C3K: Float32 = 5.597644988851731


def sh_num_bases(degree: Int) -> Int:
    return (degree + 1) * (degree + 1)
