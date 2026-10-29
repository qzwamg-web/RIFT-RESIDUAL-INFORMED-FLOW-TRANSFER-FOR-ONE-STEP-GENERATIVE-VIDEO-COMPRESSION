# 最终测试协议

- 三数据集：HEVC-B 5、MCL-JCV 30、UVG 7；四QP，共每分辨率168点。
- 首96真实帧；独立GOP25/25/25/21，末尾21帧直接编码成T6 latent，不复制尾帧。
- 1080P：1920×1080 BT.709 RGB→reflect到1920×1088→重建裁回1080。
- 720P：原1080RGB中心裁1280×736→重建中心裁1280×720。没有resize。
- rate：实际rANS帧payload比特数 / (实际编码帧数×编码高×编码宽)。历史结果同时保存original-pixel归一化项，论文默认使用actual_bpp。
- PSNR、LPIPS-VGG、LPIPS-Alex、DISTS按原始float重建逐帧计算；预览MP4不参与评分。
- FloLPIPS覆盖95对相邻帧（含跨GOP边界），保留undefined/finite-pair诊断，不能忽略无定义样本再称为完整官方指标。
- FID：原网格及偏移网格256×256 patch；每数据集池化特征统计再计算FID。保留每视频FP64 moments以便复核，不能平均视频FID。
- 每QP原子写盘；恢复时校验checkpoint、视频、协议及完整指标/视频文件。
- `scripts/benchmark.py`分开测VAE encode、codec encode/decode、条件、DiT/heads、VAE decode。报告warmup后中位数，单段Beauty25帧，不冒充全数据集速度。

既有headless完整42视频测试仅包含PSNR、LPIPS-VGG、DISTS、LPIPS-Alex及patch FID；未将其他实验的FloLPIPS混入该消融。
