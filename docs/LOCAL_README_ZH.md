# RIFT — C512

本目录是以已选定的 **C512** 为起点的训练与推理框架。默认加载 C512 完整的 codec、恢复连接头、DiT 适配器及 rank-8 LoRA。运行中的可选研究分支已收敛到 C512；本次联合微调的无连接头消融保留独立入口。

## Source release

This branch contains the final RIFT C512 runtime and the retained headless
ablation entry points. Model weights, datasets, videos, credentials, local
research experiments and historical Git objects are not distributed here.
See [asset layout](docs/ASSETS.md) and [third-party sources](third_party/VERSIONS.md).
The checkpoint remains the original C512 numerical anchor; the newer UF-L/Wan
research framework is a separate project.

## 目录

```text
raqvc/backbone/       冻结 Wan、有效条件适配器、VAE
raqvc/codec/          C512 固定 codec 与区域量化
raqvc/training/       C512 恢复端、损失与参数组、独立 headless 消融
raqvc/data/           对齐原始 RGB 的 256/512 latent 数据读取
raqvc/metrics/        PSNR、LPIPS、DISTS、FloLPIPS、patch FID
scripts/final.py     统一预检、smoke、训练、测试、计时入口
configs/             C512 codec 参数快照、42 视频列表
third_party/         必需的 DCVC/Wan 源码与许可证
```

全部实验资产归档到 `/qzwang/data/raqvc_artifacts/C512_FINAL_v1`：

- `model/`：C512 原始完整 bundle、合并 LoRA 的完整模型、本次 headless 最终权重与优化器状态、加载所需不可变基座及 VAE。
- `datasets/{256,512}/`：训练索引、原始视频名称列表、tar 分片列表。实际数据仍在 `/qzwang/data/OpenVidHD`；两个分辨率均为 **83,368 视频 / 166,736 窗口**。
- `results/C512_first96/formal/`：720P 与 1080P 全部 **336 个视频×QP结果**、重建视频、逐帧指标、FID统计、数据集汇总。
- `results/headless_joint/formal/evaluation/`：本次 2048-step 无连接头联合微调的 42 视频测试。
- `results/baselines_720/`、`results/baselines_1080_flolpips/`、`references/`：本地基线结果和论文读图数据/来源。
- `paper_revision_*`、`rd_revision_v6`、`subjective*`、`comparison_demo*`：已完成论文图表、可编辑PPT与演示视频；其余旧目录的PPT集中在 `preservation/presentations/`。
- `preservation/`：模型SHA256、全部PPT索引、数据完整性、真实训练LR/BS记录、清理清单和验证结果。

## 环境

本机使用 `/root/miniconda/envs/gvc/bin/python`。`requirements-lock.txt` 保存当前环境的版本快照；`pyproject.toml` 给出基础依赖。新环境还需匹配 CUDA 的 PyTorch、DCVC 熵编码扩展、LPIPS/DISTS/FloLPIPS 权重。预检会核对本地数据与指标权重，不隐式下载。

```bash
cd /qzwang/code/iclr2027gvc/Multimodal-Generative-Coding-final
conda activate gvc
# Install the matching CUDA/PyTorch environment and metric dependencies first.
python -m pip install pybind11
(cd third_party/DCVC/src/cpp && python setup.py build_ext --inplace)
python -m pip install --no-deps -e .
python scripts/final.py preflight
```

路径可通过 `RAQVC_ARTIFACTS` 和 `RAQVC_WAN` 指定；数据清单自身记录绝对路径，迁移机器时需要同步修改清单，不能只改环境变量。

## 运行

```bash
# 单卡真实512/T7 smoke：覆盖四QP、RGB损失、所有联合参数组和保存/重载
python scripts/final.py smoke --devices 0 --output /qzwang/data/raqvc_artifacts/C512_FINAL_v1/new_runs/smoke

# 六卡C512继续微调：每卡batch=1，固定C阶段LR，余弦衰减；结束自动42视频两分辨率六指标测试
python scripts/final.py train --devices 0,1,2,3,4,5 --steps 512 --batch 1 --evaluate-after --output /qzwang/data/raqvc_artifacts/C512_FINAL_v1/new_runs/joint

# 中断续训：保留原steps、batch、卡数和checkpoint；只改action
python scripts/final.py resume --devices 0,1,2,3,4,5 --steps 512 --batch 1 --evaluate-after --output /qzwang/data/raqvc_artifacts/C512_FINAL_v1/new_runs/joint

# 独立六卡测试；重跑同一命令会核对协议并跳过完整QP
python scripts/final.py evaluate --devices 0,1,2,3,4,5 --output /qzwang/data/raqvc_artifacts/C512_FINAL_v1/new_runs/evaluation

# 单卡编解码分段计时，结果是SPEED.json
python scripts/final.py speed --devices 0 --output /qzwang/data/raqvc_artifacts/C512_FINAL_v1/new_runs/speed
```

上述训练是**从C512继续联合微调**，默认512步是新任务的起始配置，不代表重新跑完历史训练课程。原始各阶段LR、batch和实际预算见 [训练说明](docs/TRAINING.md)。当前单卡512/T7联合smoke峰值约74.4GiB；新机器先运行smoke再决定batch。

## C512主线

VAE编码 → latent codec实际熵编解码 → 固定QP时间表下的一次DiT调用 → 继承方向/增益适配 → 独立全频RMS与局部方向/gain调制 → latent残差相加 → VAE解码。

- QP：`0,16,39,63`；Wan时间：`520,640,800,960`。
- 不添加随机噪声，无动态时间头，无 `tau/1000` 输出速度乘子，无跨GOP KV缓存。
- 保留C512真正使用的QP/codec上下文、空间条件、可靠性注意力和速度适配器。它们是checkpoint行为的一部分。
- 新RMS头估计局部全频残差强度；方向分支在基准更新上作角度修正，gain控制局部幅度。源码说明见 [框架](docs/FRAMEWORK.md)。
- 默认数值复现使用 `C512_original_bundle.pt`。`C512_merged_complete.pt` 已合并LoRA并完整保留；合并前后FP32投影有历史核对，但BF16下运算顺序会改变，不能用未经端点核验的合并文件替换已报告结果。

## 已完成结果

下面是 **QP63** 的真实数据集汇总；完整四码率点见 `results/C512_first96/formal/DATASET_METRICS.csv`。LPIPS、DISTS、FID、FloLPIPS越低越好。

| Resolution | Dataset | bpp | PSNR | LPIPS-VGG | DISTS | LPIPS-Alex | Patch FID | FloLPIPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1080 | HEVC_B | 0.0062 | 26.1360 | 0.3042 | 0.0725 | 0.1689 | 26.9219 | 0.1810 |
| 1080 | UVG | 0.0053 | 28.9444 | 0.2734 | 0.0792 | 0.1437 | 25.9373 | 0.1488 |
| 1080 | MCL-JCV | 0.0054 | 28.0211 | 0.2946 | 0.0932 | 0.1564 | 15.2820 | 0.1637 |
| 720 | HEVC_B | 0.0067 | 25.6649 | 0.3039 | 0.0783 | 0.1676 | 36.5654 | 0.1828 |
| 720 | UVG | 0.0054 | 28.8629 | 0.2765 | 0.0867 | 0.1448 | 35.2644 | 0.1575 |
| 720 | MCL-JCV | 0.0059 | 27.3766 | 0.3048 | 0.1024 | 0.1611 | 22.6745 | 0.1709 |

1080P补边到1088再重建裁回1080；720P中心裁736×1280进行编码，重建后中心裁720×1280评分。统一首96个真实RGB帧，GOP为25/25/25/21；bpp分母用实际编码画布像素数。FID使用双网格256patch数据集池化；不使用整帧FID代替。详见 [测试协议](docs/TESTING.md)。

基线的实测/论文读图来源、实际码率/熵估计码率标记保留在各自结果中；不能把来源不同的数据重新标成全部本地实测。

## 清理验证

核心重构已用原C512输入核对四QP、256/512及T6/T7，实际码率、解码latent与恢复latent逐值一致。另执行真实RGB联合训练smoke、保存重载、无连接头权重对照、720/1080视频与指标写盘smoke。详细完成状态以 `preservation/VALIDATION.json` 为准。没有重新运行完整42视频测试；已有最终结果完整保留。


## Release validation and provenance

The source snapshot is recorded in `RELEASE_MANIFEST.json`. The C512 backbone
and restoration core match the preserved source hashes. Later compact-headless
conditional hooks were removed only in this isolated release copy, with exact
hash verification; the original working directory was left intact. The current
evaluation/headless entry-point hashes are recorded separately from the original
preservation inventory. CPU import/preflight checks do not constitute a new GPU
or full42-video quality reproduction. The historical RIFT96-frame/GOP25 protocol
and its provenance are retained; do not merge those measurements with the newer
UF-L97/96 protocol under a shared reproduction label.
