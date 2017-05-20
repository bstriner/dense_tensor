# dense_tensor
Dense Tensor Layer for Keras. Supports both Keras 1 and 2. Tensor networks/second order networks.

Basically, this is like a quadratic layer if all other layers are linear.

There is an additional weight matrix V. The layer output is `xVx+xW+b` instead of simply `xW+b`.

See analysis by Socher.

* http://nlp.stanford.edu/~socherr/SocherChenManningNg_NIPS2013.pdf
* http://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf

See also Fenglei Fan, Wenxiang Cong, Ge Wang (2017) A New Type of Neurons for Machine Learning.

* https://arxiv.org/ftp/arxiv/papers/1704/1704.08362.pdf

Normal Dense Layer: `f_i = a( W_ix^T + b_i)`

Dense Tensor Layer: `f_i = a( xV_ix^T + W_ix^T + b_i)`

`DenseTensor`: same usage as Keras `Dense` Layer

## Variations

I provided several examples for different parameterizations of V, including a low-rank version of V,
a symmetric V, and V restricted to positive-definite matrices. Please explore the examples and ask any questions.

### Simple parameterization

```python
x = Input(input_dim)
layer = DenseTensor(units=units)
y = layer(x)
```

### Low-rank parameterization

```python
factorization = tensor_factorization_low_rank(q=10)
layer = DenseTensor(units=units, factorization=factorization)
```

## Comments
 
Please feel free to add issues or pull requests. I'm always interested in any improvements or issues.

## Compatibility

Travis tests a matrix including Theano, tensorflow, Python 2.7, Python 3.5, Keras 1 and Keras 2.
Code should work on most configurations but please let me know if you run into issues.
