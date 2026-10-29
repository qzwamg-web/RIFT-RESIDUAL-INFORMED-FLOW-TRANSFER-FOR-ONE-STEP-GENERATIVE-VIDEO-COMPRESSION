# C512 恢复框架

输入RGB经冻结Wan VAE得到48通道latent。固定的I/P latent codec按QP进行区域量化、熵编码及解码，得到zq和decoder可见的尺度、量化宽度、QP及latent帧角色。

`training/c512.py`严格加载完整状态，`training/restoration.py`是训练/测试共用端点。

1. `base_heads.py`：多尺度误差估计p和共享特征h；尺度校正与归一化几何融合得到s；保留继承的方向残差和token标量gain。
2. `backbone/c512_dit.py`：固定QP时间表，整段latent一次Wan调用。原codec上下文、空间条件及可靠性注意力继续生效；没有跨GOP记忆。
3. `base_restoration.py`：得到基准更新u0。QP/global及局部gain均与tau独立。
4. `residual_heads.py`：独立全频局部RMS预测rhat，I/P投影与归一化帧位置；方向分支结合邻域上下文、基准单位方向和rhat预测修正。
5. `restoration.py`：d = normalize(e0 + beta_IP (d_candidate-e0))；n=||u0||_2,channel；m=exp(log(2)tanh(ell))；a=n*m。最终数学表达为z_hat=zq+a*d。实现保留原更新/相加顺序以维持浮点数值一致性。

RMS是局部通道能量描述，不声称完整表征残差分布，也不要求不同QP残差共线。单位方向是每个latent位置的48维通道方向。

历史checkpoint中的未调用时间头和未使用预测器参数只在加载时按明确白名单过滤；其余活动权重严格核对。类和state_dict键保留兼容性，源码文件名改成实际功能名称。
