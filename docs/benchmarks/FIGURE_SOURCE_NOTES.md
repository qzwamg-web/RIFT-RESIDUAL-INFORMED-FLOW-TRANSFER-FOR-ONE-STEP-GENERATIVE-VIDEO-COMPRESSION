# RIFT 最终 R–D 图 v6

输出目录：`/qzwang/data/raqvc_artifacts/C512_FINAL_v1/rd_revision_v6`。

## 三张主图

- `RIFT_1080P_perceptual_3datasets.pdf`：HEVC-B、UVG、MCL-JCV 三行，LPIPS-VGG、LPIPS-Alex、DISTS、patch FID 四列。
- `RIFT_1080P_PSNR_3datasets.pdf`：三个数据集的 PSNR。
- `RIFT_720P_allmetrics_3datasets.pdf`：三个数据集、PSNR、LPIPS-VGG、LPIPS-Alex、DISTS、patch FID、FloLPIPS 六指标。

全部文字和曲线为矢量 PDF；PNG 仅用于预览。各方法颜色和 marker 唯一，跨图保持一致。RIFT 使用红色实线星标。

## 数据选择

按用户最新要求，1080P 外部方法统一使用所提供论文中的读图数据，RIFT 使用最终 C512 的本地42视频结果。包括 UVG 的 GLC 曲线也取论文值，避免主图不同数据集混用两种 GLC 来源。本地 GLC 的 UVG1080 实测只出现在单独的 `GLC_UVG1080_protocol_anchor.pdf` 校准图中。

720P 主图全部使用现有本地 RIFT、DCVC-RT、DCVC-FM、GLC-Video 结果，每方法42视频×4点；不包含论文720P全序列读图值。方法本身的 GOP 不要求与 RIFT 相同；现有本地 baseline 测试实际仍为25帧间隔。

1080P 论文坐标来源：`results/public_baselines_v1/data/published_rd_points.csv`。保留原始来源、图号、PDF页号、提取方式和是否位于原论文坐标轴内。排除之前用户不需要的 VVC。

### 单个可见点的连线

旧 `references()` 在绘图前过滤 `visible_inside_axis=false`，丢失了定义轴内线段的相邻点。v6直接使用经过独立marker验证的完整PDF坐标序列，先绘制折线再用坐标轴裁剪。没有合成点、外推或曲线平滑；轴外点是论文PDF内保留的坐标，不表述为原图中可见的测量点。`PLOT_AUDIT.json`记录每条曲线总顶点数、可见marker数和坐标范围。

## 本地结果核验

CPU核验868个结果：RIFT720/1080共336点，三个720 baseline共504点，GLC UVG1080共28点。15,570项检查通过，包括：

- 96帧真实编码、评分覆盖；RIFT GOP25/25/25/21。
- 码率等于比特数除以真实帧数及记录的编码画布；DCVC-RT的96个payload文件字节数吻合。
- 逐帧四项空间指标均有限，缓存均值和RESULT及数据集均值吻合。
- FloLPIPS每视频95个相邻对。
- FID每帧720P为14个patch、1080P为46个patch；同视频各方法的参考Inception特征和一致。没有重新做协方差矩阵特征值分解。

这些检查证明结果文件、汇总及参考图像一致，不证明所有论文实验配置相同。

## GLC校准与协议差异

UVG1080在三个落于共同码率区间的本地点，以log-bpp线性插值论文曲线作参照：

|指标|本地减论文值范围|
|---|---:|
|PSNR|−0.347至−0.312 dB|
|LPIPS-VGG|+0.00559至+0.00687|
|LPIPS-Alex|−0.00610至+0.00234|
|DISTS|−0.00089至+0.00175|
|FID|+0.040至+0.368|

这是共同码率比较，不按QP硬配对，也不向论文曲线范围外外推。感知指标整体接近，PSNR有系统偏差；尚未通过干预实验将其归因于特定原因。

明确记录的差异：

1. 本地 GLC 的 `fa_idx` 使用GOP内帧号；官方 `test_video.py` 使用全局帧号。本地没有官方默认每32帧清空 `ref_y` 的步骤。官方样例配置GOP96，本地使用25。因此本地是可复现的自定义协议，不应声称逐项复现官方默认配置。
2. 本地720P从原图中心裁736×1280，评分裁中心720；GLC内部再补到768高。按用户要求，GLC的bpp分母为768×1280，RT/FM/RIFT为736×1280。相同比特数采用768高分母会比736高低4.167%，未对曲线偷偷换分母。
3. RT/RIFT为实际rANS有效载荷码率；完整数据集FM/GLC为官方熵估计。此前七个主观视频的FM实际码率结果不替换完整数据集曲线。
4. 本地原图为BT709 full-range float、nearest chroma；GLC官方入口读PNG，出版参考PNG生成过程尚未核实。相同数据集名不能证明像素完全相同。
5. RIFT1080 reflect补到1088，评分裁1080，bpp按1088；论文基线保留论文自己的码率数值。FID使用256双网格、偏移128、2048维、数据集池化统计。
6. 本地完整GLC1080校准仅覆盖UVG；HEVC-B/MCL-JCV已有主观子集不能作为完整数据集校准。

本轮不修改训练/推理/评测端点，不重命名或覆盖历史结果，也不启动GPU任务。新图按用户接受的来源口径制作；不将“GLC接近”提升为全部方法协议完全一致的证明。

## 复现绘图

```bash
cd /qzwang/code/iclr2027gvc/Multimodal-Generative-Coding-final
OPENBLAS_NUM_THREADS=2 /root/miniconda/envs/gvc/bin/python scripts/agent/audit_rift_rd_protocol.py
OPENBLAS_NUM_THREADS=2 /root/miniconda/envs/gvc/bin/python scripts/agent/plot_rift_paper_rd_v6.py
```

`LOCAL_POINTS.csv`、`PLOTTED_POINTS.csv`、`GLC_UVG1080_ANCHOR.csv`、`PROTOCOL_AUDIT.json`和`PLOT_AUDIT.json`保存完整可追溯数据。
