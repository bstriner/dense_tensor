# dense_tensor
Dense Tensor Layer for Keras

http://nlp.stanford.edu/~socherr/SocherChenManningNg_NIPS2013.pdf
http://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf

Normal Dense Layer: f_i = a( W_ix^T + b_i)
Dense Tensor Layer: f_i = a( xV_ix^T + W_ix^T + b_i)

"DenseTensor": same usage as Keras "Dense" Layer but has additional argument "V_regularizer".