# 训练数据与真实课程记录

## 数据

256和512索引均含83,368源视频、166,736窗口；每个窗口25个RGB帧、T7 latent。`datasets/<resolution>/videos.txt`是视频名单，`shards.txt`是缓存列表，`index.json`含窗口offset、空间变换和源manifest哈希。原始视频、tar分片和CACHE_MANIFEST都保留在OpenVidHD原目录。

## C512后期课程

|阶段|分辨率|实际预算|每卡batch|GPU|
|---|---:|---:|---:|---:|
|A|512|4096|1|6|
|B缩短续训|256|2304|3|6|
|C（最终C512）|512|512|1|6|
|本次无连接头联合消融|512|2048|1|6|

以上是运行记录，论文另设的计划预算不替代实际日志。精确的初始checkpoint、优化器状态、world/batch和调度信息见 `preservation/training_records/`；全部原始阶段配置已保存，包含codec先导训练及Stage3父模型。

|参数组|A LR|B LR|C LR|
|---|---:|---:|---:|
|codec backbone|冻结|2e-6|1e-6|
|regional quantizer|冻结|1e-5|5e-6|
|full-band strength|2e-5|3e-5|2e-5|
|gain|3e-5|3e-5|2e-5|
|direction|1e-5|1e-5|1e-5|
|inherited adapters|冻结|5e-6|3e-6|
|rank-8 LoRA|5e-6|1e-5|5e-6|

`PHASE_LRS`及准确的参数名选择逻辑在 `raqvc/training/c512.py`。统一入口以C阶段继续训练，AdamW weight_decay=0.01、梯度裁剪1.0、warmup后余弦降至初始LR的10%。四QP循环，pixel监督每步启用。RGB LPIPS/DISTS/L1、时域项、latent项、RMS/方向辅助项、rate项和codec-only RGB辅助项均保留原C512公式。

本次headless使用codec=1e-6、regional=5e-6、LoRA=3e-5、继承context adapter=3e-6。其恢复端没有连接头，不能用正常C512 evaluator载入。入口为 `scripts/train_headless.py` / `scripts/evaluate_headless.py`。

新训练使用新AdamW。原C512及headless的优化器快照另行保留；其记录中的旧路径是来源信息。`final.py resume`用于新框架自己生成的断点，不把历史已完成阶段冒充新训练的断点。
