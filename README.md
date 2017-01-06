# dense_tensor
Dense Tensor Layer for Keras. These are some extensions of an idea mentioned by Socher.

Basically, this is like a quadratic layer if all other layers are linear.

There is an additional weight matrix V. The layer output is xVx+xW+b instead of simply xW+b.

* http://nlp.stanford.edu/~socherr/SocherChenManningNg_NIPS2013.pdf
* http://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf

Normal Dense Layer: f_i = a( W_ix^T + b_i)

Dense Tensor Layer: f_i = a( xV_ix^T + W_ix^T + b_i)

"DenseTensor": same usage as Keras "Dense" Layer but has additional argument "V_regularizer".

##Variations

I provided many different examples for different parameterizations of V, including a low-rank version of V,
 a symmetric V, and V restricted to positive-definite matrices. Please explore the examples and ask any questions.
 
 ##Comments
 
 Please feel free to add issues or pull requests. I'm always interested in any improvements or issues.