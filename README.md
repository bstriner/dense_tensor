# dense_tensor
Dense Tensor Layer for Keras. Tensor networks/second order networks.

Basically, this is like a quadratic layer if all other layers are linear.

There is an additional weight matrix V. The layer output is xVx+xW+b instead of simply xW+b.

See analysis by Socher.

* http://nlp.stanford.edu/~socherr/SocherChenManningNg_NIPS2013.pdf
* http://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf

See also Fenglei Fan, Wenxiang Cong, Ge Wang (2017) A New Type of Neurons for Machine Learning.

* https://arxiv.org/ftp/arxiv/papers/1704/1704.08362.pdf

Normal Dense Layer: f_i = a( W_ix^T + b_i)

Dense Tensor Layer: f_i = a( xV_ix^T + W_ix^T + b_i)

"DenseTensor": same usage as Keras "Dense" Layer but has additional argument "V_regularizer".

##Variations

I provided many different examples for different parameterizations of V, including a low-rank version of V,
 a symmetric V, and V restricted to positive-definite matrices. Please explore the examples and ask any questions.
 
##Comments
 
 Please feel free to add issues or pull requests. I'm always interested in any improvements or issues.