"""MojoSplat: 3D Gaussian Splatting forward rendering in pure Mojo.

Pipeline: projection -> binning (count, scan, write, radix sort, tile offsets)
-> rasterization. See `render.mojo` for the high-level `Renderer`.
"""
