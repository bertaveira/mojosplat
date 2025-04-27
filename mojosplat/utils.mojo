from layout import Layout, LayoutTensor


alias dynamic_layout_1d = Layout.row_major(1).make_shape_unknown()
alias dynamic_layout_2d = Layout.row_major(1,1).make_shape_unknown()
alias dynamic_layout_3d = Layout.row_major(1,1,1).make_shape_unknown()


alias Tensor1D = LayoutTensor[DType.float32, dynamic_layout_1d, MutableAnyOrigin]
alias Tensor2D = LayoutTensor[DType.float32, dynamic_layout_2d, MutableAnyOrigin]
alias Tensor3D = LayoutTensor[DType.float32, dynamic_layout_3d, MutableAnyOrigin]


