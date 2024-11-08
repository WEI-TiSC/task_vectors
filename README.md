# Experiment

## White Box
One model + opposite known to test
`e.g. MNIST and know opposite: SVHN, train MNIST_SVHN_perm and test`

## Black Box
One model + another dataset, test other datasets

`e.g . MNIST model use SVHN params for permuatation, other datasets for test` 

**Note: Models in White Box setting could be directly used!**
In fact, it could be considered as an extend to white box, which is, what if 
the model is combined by opposite 2 while you permuted using opposite 1?