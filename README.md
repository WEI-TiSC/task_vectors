# Experiment

``Generate Permuted Models: source run_permutation_vit-b-32.sh``

``Run Evaluations: source run_permute_evaluation``

**实验设计**：
- 白盒（知道确定的对方，permute）-> 灰盒（知道一个对方，permute并测不确定对方）-> 黑盒（只知道自己的预训练节点并permute）
  - 白盒阶段测试组数：8(自己模型)*7(可能的对方模型)*2(不同的`\lambda`)=112组；
  - 灰盒阶段测试组数：8(自己模型)*6(未知的对方模型)*2(不同`\lambda`,也可以先只测`\lambda=0.3`)；
    - 直接用白盒的permute模型（随便选一个）然后跑TA permuted_vic, permuted_fr两个因为benign数据已在白盒测过；

  - 黑盒阶段每个模型和预训练节点permute后测与另外7个任务性能（如果假设`\lambda=0.3, 0.8`两个情况则测：7*8*2=112组（同白盒数量）

**近期安排**：
周六：追加另外三个数据集并跑模型。（放服务器上跑，服务器下载teamviewer传输数据！）