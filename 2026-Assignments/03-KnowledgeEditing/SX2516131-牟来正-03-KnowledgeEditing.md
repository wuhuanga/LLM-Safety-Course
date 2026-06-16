# Task1

### 基础环境搭建与基线测试

![image-20260525144221816](C:\Users\4090\AppData\Roaming\Typora\typora-user-images\image-20260525144221816.png)

![image-20260525144310292](C:\Users\4090\AppData\Roaming\Typora\typora-user-images\image-20260525144310292.png)

**从模型的回答可以看出，答案是过时的知识或者就是因存在知识盲区而回答的错误答案**。

# Task2

### 单条事实编辑实践 ROME

![image-20260525144448092](C:\Users\4090\AppData\Roaming\Typora\typora-user-images\image-20260525144448092.png)

![image-20260525144633056](C:\Users\4090\AppData\Roaming\Typora\typora-user-images\image-20260525144633056.png)

![image-20260525144836863](C:\Users\4090\AppData\Roaming\Typora\typora-user-images\image-20260525144836863.png)

# Task3

### 批量知识编辑实践 MEMIT

![image-20260525154545727](C:\Users\4090\AppData\Roaming\Typora\typora-user-images\image-20260525154545727.png)

![image-20260525163302231](C:\Users\4090\AppData\Roaming\Typora\typora-user-images\image-20260525163302231.png)

![image-20260525163424404](C:\Users\4090\AppData\Roaming\Typora\typora-user-images\image-20260525163424404.png)

# Task4

![image-20260525163614127](C:\Users\4090\AppData\Roaming\Typora\typora-user-images\image-20260525163614127.png)

| 方法  |     ES |     PS |     NS |
| ----- | -----: | -----: | -----: |
| ROME  | 95.00% | 85.50% | 69.67% |
| MEMIT | 45.19% | 44.89% | 27.76% |

### MEMIT 批量编辑结果分析

从实验结果来看，MEMIT 在 500 条知识的批量编辑任务中取得了 `ES=45.19%`、`PS=44.89%`、`NS=27.76%`。相比 ROME 方法的 `ES=95.00%`、`PS=85.50%`、`NS=69.67%`，MEMIT 的整体表现明显较低。

造成这一结果的原因主要有以下几点。首先，ROME 是逐条进行知识编辑，并且本实验中将 ROME 的 loss 阈值设置得较低，同时允许更多迭代次数，因此每条知识能够被更充分地优化，编辑成功率较高。相比之下，MEMIT 是一次性对多条知识进行批量注入，需要同时兼顾大量编辑目标，不同样本之间可能存在相互干扰，因此单条知识的编辑效果会被削弱。

其次，本实验中为了加快 MEMIT 的运行速度，将 `mom2_n_samples` 设置为 `100`。该参数用于控制计算二阶矩 / 协方差统计量时使用的样本数量。样本数越大，统计估计越稳定，MEMIT 在写入知识时越能减少对原模型能力的破坏；但样本数越小，计算速度越快，同时统计量会更粗糙。因此，`mom2_n_samples=100` 虽然显著缩短了运行时间，但也可能导致协方差估计不充分，从而影响编辑成功率和邻域保持能力。

从指标上看，MEMIT 的 `ES=45.19%` 表明约有不到一半的编辑请求能够在原始问题上成功输出目标答案；`PS=44.89%` 与 ES 接近，说明对于成功编辑的部分知识，模型在改写问法上的泛化能力有限；`NS=27.76%` 较低，说明批量编辑对邻域知识造成了较明显的干扰，模型在保留未编辑知识方面表现较弱。

综上，MEMIT 的优势在于能够支持批量知识编辑，适合一次性注入大量知识；但在本实验设置下，由于使用了较小规模模型，并且为了加快运行将 `mom2_n_samples` 降低到 `100`，导致协方差统计量估计不够充分，最终表现为编辑成功率、泛化能力和邻域保持能力都较低。后续如果希望提升 MEMIT 效果，可以增大 `mom2_n_samples`，例如设置为 `1000` 或更高，同时适当减少单次批量编辑数量，分批进行编辑，以在运行效率和编辑效果之间取得更好的平衡。