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

## Reproduce

``Generate Permuted Models: source run_permutation_vit-b-32.sh``

``Run Evaluations: source run_permute_evaluation``

- 后续跑black box只需要跑TA permuted_vic, permuted_fr两个就行！因为其他数据都有了！

周六：追加另外三个数据集并跑模型。（放服务器上跑，服务器下载teamviewer传输数据！）