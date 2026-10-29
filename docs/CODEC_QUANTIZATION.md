# Headless codec量化网络核对

## 结论

不是一个单一的“逐通道逐位置scale”。熵分布尺度sigma、动态量化q_enc/q_dec、区域量化宽度w是不同张量。操作对象是codec分析变换后的y，不是直接对48通道VAE latent逐元素预测一个独立scale。

- 区域宽度w：I/P各一个共享MLP(3→32→1)，各161参数，总322参数；输入为组内平均log|sigma|、组内平均mu²、QP/63。每16通道共享宽度，每个空间位置单独预测，同一个MLP跨组/位置使用；没有group embedding。各帧、各熵编码pass会重新估计。
- 基础width在四QP均为1；w=exp(log(2)*tanh(MLP(features)))，范围[0.5,2]。因此QP区别来自MLP输入以及其他动态/查表缩放，不是固定基础width表的差异。
- I动态q_enc和q_dec：各[B,1,Hy,Wy]，按位置变化、通道共享；两者分别预测，不应当视为必然互为倒数。I的符号是round((y*q_enc-mu)/w)，重建为(w*symbol+mu)*q_dec；在y编码坐标中分辨间隔是w/q_enc，重建间隔是w*q_dec。
- P动态q_dec：本身[B,128,Hy,Wy]。原始q_dec先至少0.5，再乘QP/type/channel查表因子exp(2.5*tanh(logit))并至少0.5；查表因子无空间维。P符号round((y/q_dec-mu)/w)，重建(w*symbol+mu)*q_dec。最终有效步长q_dec*w逐通道逐位置变化，但w自身仍按16通道共享。
- sigma是熵概率尺度，逐通道逐位置预测，熵编码使用sigma/w；不是直接等同于步长，也不是独立恢复RMS头。
- 对外导出的width map经过通道/编码mask聚合，是单通道空间图；它不能展示内部所有通道的量化差异。

## 本轮真实探针

使用旧headless、512/T7固定样本、QP63（仅一个样本，不代表全数据集），在实际熵编解码中记录：

| 张量 | shape | 观测范围 |
|---|---|---|
| I区域w | [1,320,16,16] | 1.03504–1.20097 |
| P区域w | [1,128,16,16] | 1.00665–1.01510 |
| I q_enc | [1,1,16,16] | 0.50145–0.50429 |
| I q_dec | [1,1,16,16] | 1.77028–1.83608 |
| P有效q_dec | [1,128,16,16] | 5.87338–18.56103 |

I为20组，P为8组。实际w组内最大差均为0；空间标准差非0。此样本P区域调制弱，主要差异在q_dec，不能据此断言区域网络对所有视频无效。完整四QP/尺寸采样记录见CODEC_QUANTIZATION.json。

源码：raqvc/codec/task938_regional_quantizer.py、rt_two_codec.py中的_task938_effective_width、_separate_i_entropy_prior、_effective_p_q_dec，以及DCVC common_model.py separate_prior。
