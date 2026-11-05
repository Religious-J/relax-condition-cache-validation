# Relax No.17 条件缓存验证材料

本仓库收录 [Relax #544](https://github.com/redai-studio/Relax/pull/544)、[#545](https://github.com/redai-studio/Relax/pull/545) 的验收工具、必要测试和公开实验记录。训练实现保留在 Relax；这里不复制整个框架、不提供另一套训练器，也不会自动提交 GPU 任务。

**所有 GPU 数据均绑定下面列出的历史源码。最新一批测量来自 `bd07ff5`，不能据此声称后续验收代码外移或监控精简后的新提交已通过 GPU 性能验收。**

## 无需 GPU 检查记录

Python 3.9 及以上即可运行，不需要 Torch、Ray、模型或数据：

```bash
python3 verify.py
python3 recompute.py
python3 recompute.py --output /tmp/no17-recomputed.json
```

`verify.py` 检查公开文件的大小、SHA-256、gzip 内容以及迁移工具/测试的哈希；`recompute.py` 解压到临时目录，复用原公共解析器，根据所有 rank 的事件与 Ray `job.json` 重算吞吐、波动、完整耗时、累计回本和资源观察值，结束后释放临时目录。

**精度 JSON 是当时运行产生的逐参数比较报告。查看或检查这些 JSON 不能重新计算原始梯度、参数更新或缓存张量的差异。** 完整权重、初始状态张量、训练 checkpoint、模型和数据图片不在本仓库。哈希检查用于验证材料内部完整性，不等同于独立证明实验真实性。

## 实验范围与结果

| 证据目录 / 源码 | 负载 | 在线 → 缓存（samples/s） | 稳态收益 | 重复次数 |
| --- | --- | ---: | ---: | --- |
| [sensenova-a983b3c](evidence/sensenova-a983b3c) / [`a983b3c`](https://github.com/Religious-J/Relax/commit/a983b3cfb5880616d8dcdcde204989fd371b1995) | SenseNova / MagicBrush128 | 5.347 ± 0.025 → 6.208 ± 0.039 | +16.10% | 2 对，AB/BA |
| [qwen-9179512](evidence/qwen-9179512) / [`9179512`](https://github.com/Religious-J/Relax/commit/91795121a565079a1a5ad3024b6a44c6c497814e) | Qwen-Image / COCO128 | 10.503 ± 0.141 → 11.178 ± 0.129 | +6.43% | 2 对，AB/BA |
| [review-bd07ff5](evidence/review-bd07ff5) / [`bd07ff5`](https://github.com/Religious-J/Relax/commit/bd07ff5e8146207b9319c6b42be426c058be0a82) | SenseNova / MagicBrush128 | 5.302 → 6.298 | +18.78% | 1 对回归 |
| 同上 | Qwen-Image / COCO128 | 10.288 → 11.110 | +7.99% | 1 对回归 |

`±` 为样本标准差。`bd07ff5` 的 n=1 没有运行间标准差，不能替代重复性能验收。旧 SenseNova/Qwen 独立低复用结果分别为 **−17.55% / −0.31%**（各模式 n=1）；这些结果保留各自旧提交归属，`bd07ff5` 没有重测低复用。

全部为单机 8×H20 上的实际 Ray SFT。每步使用最慢 rank 的完整训练循环时间，全局有效样本仅计一次，包含数据准备、条件装载、前向、反向与优化器更新。当前收录 20 个成功任务：SenseNova 历史验收 7 个、Qwen 恢复基线后验收 7 个、review 修复回归 6 个。

- SenseNova 使用 `sensenova/SenseNova-U1.5-8B-MoT` revision `19bc874ef6ffc97fda9837b40fc1d1301806158a`、MagicBrush revision `6c254560879df7851233ea0fd2ae2a62764d0ef2`；固定首个 train parquet 前 128 个物理行，保留原顺序和图像。有效 batch 8、每卡 micro batch 1、`max_pixels=1048576`，64 步/512 样本；前 16 步为预先声明的冷窗口。
- Qwen 使用 `Qwen/Qwen-Image` revision `75e0b4be04f60ec59a75f475837eced720f823b6`、COCO2017 revision `036f3f8291db64d17faad9b09e59dd30bb65c4d7`；固定首个 train parquet 前 128 个物理行，每图第一条原始 caption。有效 batch 16、每卡 micro batch 1、512×512，64 步/1,024 样本；前 8 步为冷窗口。
- LoRA rank/alpha 32，seed 42，BF16 参数、FP32 归约和主权重；AdamW 参数、编码器放置、已有优化及样本顺序均记录在各组 `plan.json`、各任务 `request.json` 与 init 事件中。数值验证与性能任务分开运行，数值验证 warmup 0，性能 warmup 200。

四份逐参数数值报告均记录 loss、完整有效 batch 累积并归约后的梯度与实际 FP32 更新量 `Δθ` 精确一致，缓存实际命中分支得到验证；SenseNova 为 588 个梯度/588 个更新量张量，Qwen 为 960/960，误差均为 0，更新非全零。详细报告为每个 verify 任务中的 `conditioning-verification.json.gz`。

### review 修复后的完整耗时

以下仅对应 `bd07ff5` 的单次实验，单位为秒，单元格为在线 → 缓存。

| 计时范围 | SenseNova | Qwen-Image |
| --- | ---: | ---: |
| 首轮冷窗口 | 36.543 → 47.056 | 31.213 → 34.287 |
| 后续稳态窗口 | 72.425 → 60.972 | 87.093 → 80.646 |
| 含冷构建的完整训练循环 | 108.969 → 108.028（+0.87%） | 118.305 → 114.934（+2.93%） |
| 含启动/退出的完整 Ray job | 281.424 → 279.327（+0.75%） | 307.763 → 309.567（−0.58%） |

这里“冷”表示新缓存 namespace，未清空 OS page cache；首轮窗口不能视为纯写缓存耗时。累计训练内耗时从 SenseNova 第 60 步、Qwen 第 37 步起节省持续非负，分别为 480/592 个样本，排除启动/退出。Qwen 此次完整任务仍未加速。

### 资源、生命周期与共享价值边界

`resources.jsonl.gz` 保留外部 15 秒采样，`resource_after.json` 保留任务结束后的资源观察。复算输出区分每 rank 运行期统计与完整 job 窗口的外部观测最大值；CPU RSS 可能重复计共享页，采样不保证捕获瞬时峰值。

`bd07ff5` 稳态共享 payload 分别固定为 SenseNova **19,575,839,120 B**、Qwen **48,541,696 B**；节点 payload/写入暂存上限分别为19/19 GiB、1/1 GiB，SenseNova 条件预取另限512 MiB/rank。六个任务结束后缓存 namespace/payload 归零，GPU 空闲；每 rank 的 step/close 待完成复制数量与字节为0。正常结束证据不能代替异常或取消验证；本批没有真实GPU取消实验。

**这些吞吐结果不证明节点共享缓存优于 rank-local 缓存。** 现有事件没有 producer rank 与缓存 key 的关联，无法据样本ID判断某次实际命中是否跨 rank，也不能证明同样负载下减少了多少重复存储。原有 payload 容量、读者租约及跨进程生命周期 CPU 测试仍属于生产实现的回归测试；它们验证正确性，不构成共享架构的性能消融。证明共享收益需要额外记录实际跨 rank 复用/去重证据，并在相同负载、容量和设备下对照，当前材料不作这项结论。

### 可用于下一步优化的已有成本观察

从 `bd07ff5` 缓存路径最后冷步到最后训练步的累计 `cache_stats` 作差，以下为 **8 rank 平均 host 秒数**：

| 诊断计数 | SenseNova | Qwen-Image |
| --- | ---: | ---: |
| checksum | 6.539 | 0.041 |
| 等锁 | 0.014 | 0.449 |
| 持锁 | 0.068 | 0.114 |
| mmap | 0.109 | 0.024 |
| load（含其内部阶段） | 6.713 | 0.571 |

SenseNova 的大payload校验、Qwen的小payload下锁与索引操作值得分别调查。预取可能跨越冷/稳态边界，阶段互相重叠，`load`也包含子阶段；**不能相加、除以端到端时间后当作精确瓶颈占比，更不能直接作为优化收益**。现有数据不足以宣称key构造、失效检查或共享存储架构本身是主要瓶颈。

## 材料完整性、脱敏与选择规则

- [MANIFEST.json](MANIFEST.json) 列出每份文件的原始 SHA-256/字节数、公开文件 SHA-256/字节数、gzip 解压后哈希，以及逐文件脱敏类别。内网IP、个人/机器路径前缀和Ray节点标识替换为占位符；数值、样本顺序、参数名称及误差指标保持不变。脱敏后的日志不能称为原始字节。
- 原始 plan/request 中的哈希继续指向原始内容。脱敏会改变这些文件自身的哈希，不能利用脱敏副本重算并证明原始 plan/source-envelope 完全一致。原始文件保存在私有归档；持有者可用 `python3 verify.py --original-artifacts /path/to/artifacts` 逐份校验原始哈希。
- SenseNova 组只纳入原计划中的全部7个SenseNova任务；同批已被后续Qwen恢复方案替代的旧Qwen任务未重复收录。Qwen组完整纳入最终7个任务，并保留观察进程中断后的补录说明；未重新训练或替换这次低复用结果。
- review组完整纳入r02的6个成功任务。此前r01在模型加载前因不完整shallow Git元数据失败，没有训练事件；未计入吞吐。此选择及源计划任务数量在manifest说明，未筛选最快重复。
- 逐run重复的harness和空patch不复制；manifest记录三脚本哈希及全部空patch事实。完整源代码以Git提交为准。大件模型、图片、bundle与checkpoint不上传。

## 工具与测试迁移

[MIGRATION.json](MIGRATION.json) 记录从 `bd07ff5` 移出的两个纯验收模块、两份测试及原说明文件，保留源路径和原始哈希。两个测试只调整独立模块的导入路径。旧Qwen整合测试保留源路径/哈希/链接，未重复存入本仓库。产品缓存、容量、生命周期、RNG等回归继续留在Relax，不在这里复制。

已有带 Torch/NumPy/pytest 的兼容环境可运行：

```bash
python -m pytest tests/test_benchmark_condition_cache.py tests/test_conditioning_verification.py -q
```

标准库的查看/复算路径不导入 `conditioning_verification.py`；后者执行真实数值对照时才需要Torch和NumPy。

## 在历史源码上重跑

历史复现必须使用对应提交中的生产训练路径，不能将本仓库工具覆盖安装进任意新版本Relax：

```bash
git clone https://github.com/Religious-J/Relax.git Relax-source
git -C Relax-source checkout bd07ff5e8146207b9319c6b42be426c058be0a82
cd Relax-source
```

按所选 `plan.json` 和 `request.json` 准备匹配版本的模型/数据、依赖和8卡Ray环境，并将已脱敏路径、Ray地址及输出目录替换为自己的设置。使用该固定源码已有的 `scripts/training/diffusion/run-sensenova-condition-cache.sh` 或 `run-qwen-image-sft-condition-cache.sh`，通过 `scripts/entrypoint/ray-job.sh` 提交。生产源码中的[原验证说明](https://github.com/Religious-J/Relax/blob/bd07ff5e8146207b9319c6b42be426c058be0a82/scripts/benchmarks/conditioning_cache/README.md)列出数值验证环境变量、非零更新warmup和独立性能运行方式。

后续精简后的生产代码不依赖本验收模块；若要验证新源码，应由外部测试控制在线参考与缓存状态，例如临时撤下 `model.conditioning_cache` 后恢复，不重新引入生产 `force_online` 分支。那会产生新的源码/配置/测量记录，不能沿用本仓库历史GPU结论。

## 许可证

迁移代码沿用 [Apache-2.0](LICENSE) 及原版权头；模型和数据遵循其各自许可证。本仓库不分发模型权重或数据图片。
