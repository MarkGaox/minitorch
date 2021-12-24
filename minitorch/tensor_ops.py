import numpy as np
from .tensor_data import (
    to_index,
    index_to_position,
    broadcast_index,
    shape_broadcast,
    MAX_DIMS,
)


def tensor_map(fn):
    """
    Low-level implementation of tensor map between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      broadcast. (`in_shape` must be smaller than `out_shape`).

    Args:
        fn: function from float-to-float to apply
        out (array): storage for out tensor
        out_shape (array): shape for out tensor
        out_strides (array): strides for out tensor
        in_storage (array): storage for in tensor
        in_shape (array): shape for in tensor
        in_strides (array): strides for in tensor

    Returns:
        None : Fills in `out`
    """

    def _map(out, out_shape, out_strides, in_storage, in_shape, in_strides):
        length = len(out_shape)

        def recursive_set(cur_out_index, shape_index):
            """
            Recursively treaverse the `out_storage`. For each cell, find
            its corresponding value inside the `in_storage`, map it, and
            set the value in `out_storage`.

            Args:
                cur_out_index (array): current index of `out_storage`
                shape_index (int): the index of `out_shape` we are at

            Returns:
                None: Fills in 'out'
            """
            if shape_index >= length:
                return
            for i in range(out_shape[shape_index]):
                cur_out_index[shape_index] = i

                # Find the index for input tensor that has shape of in_shape
                cur_in_index = [0] * length
                broadcast_index(cur_out_index, out_shape, in_shape, cur_in_index)
                cur_in_index = [i for i in cur_in_index if i != -1]

                # Find input and output storage position, then set the value of output storage
                in_storage_position = index_to_position(cur_in_index, in_strides)
                out_storage_position = index_to_position(cur_out_index, out_strides)
                out[out_storage_position] = fn(in_storage[in_storage_position])

                recursive_set(cur_out_index.copy(), shape_index + 1)

        recursive_set([0] * length, 0)

    return _map


def map(fn):
    """
    Higher-order tensor map function ::

      fn_map = map(fn)
      fn_map(a, out)
      out

    Simple version::

        for i:
            for j:
                out[i, j] = fn(a[i, j])

    Broadcasted version (`a` might be smaller than `out`) ::

        for i:
            for j:
                out[i, j] = fn(a[i, 0])

    Args:
        fn: function from float-to-float to apply.
        a (:class:`TensorData`): tensor to map over
        out (:class:`TensorData`): optional, tensor data to fill in,
               should broadcast with `a`

    Returns:
        :class:`TensorData` : new tensor data
    """

    f = tensor_map(fn)

    def ret(a, out=None):
        if out is None:
            out = a.zeros(a.shape)
        f(*out.tuple(), *a.tuple())
        return out

    return ret


def tensor_zip(fn):
    """
    Low-level implementation of tensor zip between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `out_shape`
      and `a_shape` are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `a_shape`
      and `b_shape` broadcast to `out_shape`.

    Args:
        fn: function mapping two floats to float to apply
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        b_storage (array): storage for `b` tensor
        b_shape (array): shape for `b` tensor
        b_strides (array): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """

    def _zip(
        out,
        out_shape,
        out_strides,
        a_storage,
        a_shape,
        a_strides,
        b_storage,
        b_shape,
        b_strides,
    ):
        length = len(out_shape)

        def recursive_set(cur_out_index, shape_index):
            """
            Recursively treaverse the `out_storage`. For each cell, find
            its corresponding value inside the `a_storage` and `b_storage`,
            zip them, and set the result value in `out_storage`.

            Args:
                cur_out_index (array): current index of `out_storage`
                shape_index (int): the index of `out_shape` we are at

            Returns:
                None: Fills in 'out'
            """
            if shape_index >= length:
                return
            for i in range(out_shape[shape_index]):
                cur_out_index[shape_index] = i

                # Find the index for input tensor `a` that has shape of a_shape
                cur_a_index = [0] * length
                broadcast_index(cur_out_index, out_shape, a_shape, cur_a_index)
                cur_a_index = [i for i in cur_a_index if i != -1]

                # Find the index for input tensor `b` that has shape of b_shape
                cur_b_index = [0] * length
                broadcast_index(cur_out_index, out_shape, b_shape, cur_b_index)
                cur_b_index = [i for i in cur_b_index if i != -1]

                # Find a, b and output storage position, then set the value of output storage
                a_storage_position = index_to_position(cur_a_index, a_strides)
                b_storage_position = index_to_position(cur_b_index, b_strides)
                out_storage_position = index_to_position(cur_out_index, out_strides)
                out[out_storage_position] = fn(a_storage[a_storage_position], b_storage[b_storage_position])

                recursive_set(cur_out_index.copy(), shape_index + 1)

        recursive_set([0] * length, 0)

    return _zip


def zip(fn):
    """
    Higher-order tensor zip function ::

      fn_zip = zip(fn)
      out = fn_zip(a, b)

    Simple version ::

        for i:
            for j:
                out[i, j] = fn(a[i, j], b[i, j])

    Broadcasted version (`a` and `b` might be smaller than `out`) ::

        for i:
            for j:
                out[i, j] = fn(a[i, 0], b[0, j])


    Args:
        fn: function from two floats-to-float to apply
        a (:class:`TensorData`): tensor to zip over
        b (:class:`TensorData`): tensor to zip over

    Returns:
        :class:`TensorData` : new tensor data
    """

    f = tensor_zip(fn)

    def ret(a, b):
        if a.shape != b.shape:
            c_shape = shape_broadcast(a.shape, b.shape)
        else:
            c_shape = a.shape
        out = a.zeros(c_shape)
        f(*out.tuple(), *a.tuple(), *b.tuple())
        return out

    return ret


def tensor_reduce(fn):
    """
    Low-level implementation of tensor reduce.

    * `out_shape` will be the same as `a_shape`
       except with `reduce_dim` turned to size `1`

    Args:
        fn: reduction function mapping two floats to float
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        reduce_dim (int): dimension to reduce out

    Returns:
        None : Fills in `out`
    """

    def _reduce(out, out_shape, out_strides, a_storage, a_shape, a_strides, reduce_dim):
        length = len(a_shape)

        def recursive_set(cur_a_index, shape_index):
            """
            Recursively treaverse the `a_storage`. Reduce the dimensionaility of `a_storage`
            by applying the reduce function, and set the value in `out_storage`.

            Args:
                cur_a_index (array): current index of `a_storage`
                shape_index (int): the index of `a_shape` we are at

            Returns:
                None: Fills in 'out'
            """
            if shape_index >= length:
                return
            if shape_index == 0:
                for_length = range(a_shape[shape_index])
            else:
                for_length = range(1, a_shape[shape_index])
            for i in for_length:
                cur_a_index[shape_index] = i

                # Find the storage position index for tensor 'a'
                a_storage_position = index_to_position(cur_a_index, a_strides)

                if cur_a_index[reduce_dim] == 0:
                    # If index of reduced dimension is 0, directly assign value to the `out` storage
                    out_storage_position = index_to_position(cur_a_index, out_strides)
                    out[out_storage_position] = a_storage[a_storage_position]
                else:
                    # Otherwise, apply `fn` to reduce the dimension
                    cur_out_index = cur_a_index.copy()
                    cur_out_index[reduce_dim] = 0
                    out_storage_position = index_to_position(cur_out_index, out_strides)
                    out[out_storage_position] = fn(out[out_storage_position], a_storage[a_storage_position])

                recursive_set(cur_a_index.copy(), shape_index + 1)

        recursive_set([0] * length, 0)

    return _reduce


def reduce(fn, start=0.0):
    """
    Higher-order tensor reduce function. ::

      fn_reduce = reduce(fn)
      out = fn_reduce(a, dim)

    Simple version ::

        for j:
            out[1, j] = start
            for i:
                out[1, j] = fn(out[1, j], a[i, j])


    Args:
        fn: function from two floats-to-float to apply
        a (:class:`TensorData`): tensor to reduce over
        dim (int): int of dim to reduce

    Returns:
        :class:`TensorData` : new tensor
    """
    f = tensor_reduce(fn)

    def ret(a, dim):
        out_shape = list(a.shape)
        out_shape[dim] = 1

        # Other values when not sum.
        out = a.zeros(tuple(out_shape))
        out._tensor._storage[:] = start

        f(*out.tuple(), *a.tuple(), dim)
        return out

    return ret


class TensorOps:
    map = map
    zip = zip
    reduce = reduce
