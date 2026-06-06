# Basics

Palettization compresses model weights by clustering them into a look-up table (LUT) of
centroids. Each weight value is replaced by an index into the LUT, reducing storage from
full-precision floats to a small number of bits per weight.

Weights with similar values are grouped together and represented using the value of the cluster centroid they belong to. The original weight matrix is converted to an index table in which each element points to the corresponding cluster center.

`n_bits={1,2,3,4,6,8}` are supported, where `n_bits` is the number of bits used for palettization. In other words, each weight value will be represented using `n_bits` precision, and the size of the LUT will be `2^n_bits` (see figures below for a visual explanation).

## Scalar and Vector Palettization

There are two types of palettization supported by `coreai-opt`:

- **Scalar palettization (default)**: each weight value is treated independently and the clustering process essentially involves performing K-means on one-dimensional data.
- **Vector palettization**: weight values can be treated as vectors of dimensionality `cluster_dim`. For example, with `n_bits=4` and `cluster_dim=2`, K-means will happen on 2-D data with `2^4=16` cluster centers. Hence, the resulting LUT will have a size of `16x2`. Since each tuple of weight values will be represented by 4 bits, the effective `bpw` (bits per weight) would be `4/2 = 2 bpw`. This would give a similar compression ratio to scalar palettization with `n_bits=2` (ignoring the space occupied by the LUTs themselves, which are typically much smaller compared to the weight tensors).

## Granularity

For both scalar and vector palettization, there are two modes (aka *granularities*) available:

- **per_tensor**: a single LUT is computed for the whole tensor.
- **per_grouped_channel**: multiple LUTs per weight tensor. The weight tensor is divided into groups, each with its own LUT. The number of LUTs is controlled by the `group_size` parameter. For example, a weight of shape `(1024, 2048)`, with `group_size=16`, applied on `axis=0`, will have `1024/16 = 64` LUTs, each covering a slice of weight values of shape `(16, 2048)`. This helps in reducing the error, and increasing accuracy of the palettized model.

## Visual Examples

![Scalar palettization with per-tensor granularity](palettization/images/palettization_scalar_per_tensor.svg)![Scalar palettization with per-grouped-channel granularity](palettization/images/palettization_scalar_per_grouped_channel.svg)![Vector palettization](palettization/images/palettization_vector.svg)
