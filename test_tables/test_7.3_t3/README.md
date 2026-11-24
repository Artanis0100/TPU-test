# §7.3 Table 3：SF100 TPU 整窗口活动

复用 VM 已有的 SF100 数据、精确参考和 Python 环境。测试 Q1.1、Q2.1、Q3.1、Q3.2、Q4.1，终端打印 HBM peak、Vector-ALU issue、XLU busy 和两颗 SparseCore 的 vector issue 百分比。不重新生成数据、不安装或升级依赖、不修改原作者源码。

## 在现有 VM 上启动

把本文件夹的**全部内容**解压到 `~/testtest_7.3_t3/`，确保该目录直接包含 `run_benchmark.py` 和其他 Python 文件。在 VM 中执行：

```bash
source ~/test/sectile-env/bin/activate
export PYTHONDONTWRITEBYTECODE=1
cd ~/testtest_7.3_t3

python -B run_benchmark.py \
  --artifact-root "$HOME/test/sectile-artifact-EF6C" \
  --work "$HOME/test/sectile-work" \
  --check-only
```

预检通过后运行五条查询：

```bash
python -B run_benchmark.py \
  --artifact-root "$HOME/test/sectile-artifact-EF6C" \
  --work "$HOME/test/sectile-work"
```

也可以先只测一条：

```bash
python -B run_benchmark.py --queries Q1.1
```

默认路径就是 `~/test/sectile-artifact-EF6C` 和 `~/test/sectile-work`，不依赖新文件夹所在位置或旧 `test_7.2_4a` 代码。端口默认 9999；若已被其他 profiler 使用，先确认已有任务状态，或用 `--port 10001`。请串行运行实验，避免其他 TPU 作业影响利用率。

默认过滤 XProf 转换阶段的 C++ INFO 日志（如 `serving tool`、`Creating ... threads`），保留结果表、警告、错误和 trace-loss 提示。Linux 上日志临时存于匿名内存描述符，不创建磁盘文件；发生转换异常时回放完整日志。需要诊断时加 `--verbose` 显示全部原生日志。其他平台若没有匿名内存描述符，则保留原生日志。

`--check-only` 不加载完整 SF100 列、不执行查询或硬件采样；它检查已有输入路径、Python 采集/转换接口，以及一个独立进程中的作者固定依赖、源码完整性和单芯片 TPU。它不能证明实际 RPC 会返回所有硬件计数器；真正运行仍会逐窗口校验。

## 测量方法与作者源码

- 使用作者 Figure 4a 同一套自动规划器、论文参数、`exact_int64`、TPU 结果整理、zone-map guard 和正确性验证。13 查询中的这里只选择论文 Table 3 的五条；每条用独立 TPU 进程，串行运行。
- 保留原始正确性测量，再增加作者协议规定的两次完整预热。固定同一个已编译 callable、同一组 TPU 常驻输入，连续调用并在每次调用后 `jax.block_until_ready()`。
- 循环期间不逐次 D2H、不在 CPU 解码 SQL 行。缩短扫描的计划重复执行 guard 加原生产者/最终整理器，循环结束后检查 guard 和精确结果；失败即拒绝本次采集。
- 在循环活动期间，通过本机 profiler RPC 采集 **3 个独立的 5000 ms 窗口**。不是按一次查询执行时间推算利用率，也不是每阶段利用率。
- HBM 复用作者 `steady_state_profiling` 的 193 计数器公式：32 字节读写 beats、1.75 GHz 周期频率、名义峰值 1525.5 GiB/s。其余列使用 XProf 官方 Utilization Viewer 的 achieved/peak。先算每个窗口百分比，再逐列取三个窗口的中位数。
- 复用 `reproduce.protocols.extract_accepted_counters` 对论文所需 228 项计数器的完整性检查。trace drop/buffer 警告按作者规则保留并提示，不能从硬件计数中扣减；缺失、重复、无效计数器和其他 profiler 错误会报错，不以 0 补齐。

主要参考：`reproduce/profile.py`、`reproduce/execution_guard/profile.py`、`scripts/evaluation/run_steady_state_profile.py`、`reproduce/profile_analysis.py` 和 `src/tpu_db/benchmark/steady_state_profiling.py`。

## 不落盘的实现与边界

作者原 CLI 必须写 profile 和审计文件，本目录改为内存传输：

1. `counter_capture.py` 使用相同 ProfilerService 的 `emit_xspace=True` 返回原始 XSpace 字节，仓库路径留空，不调用会保存文件的 native `trace()`。
2. `counter_metrics.py` 使用 `xspace_to_tools_data_from_byte_string` 转换内存数据，不提供实际输入文件或缓存目录。
3. `console_artifacts.py` 仅屏蔽作者准备/绑定流程的报告、HLO 和日志写入；输入与参考读取保持不变。Python 字节码和 JAX 持久编译缓存关闭。控制信号通过匿名进程管道传送，不创建 signal 文件。

这依赖现有 JAX/JAXLIB `0.10.2`、libtpu `0.0.42.1` 与 XProf 的相应接口。预检失败时先保留报错，不要直接升级现有环境。

Figure 4a 的终端版没有保存先前执行计划 descriptor，因此这里复用同样的自动规划配置，并核对本次三个窗口内计划与输入不变；无法单独证明它与此前 VM 运行保存的某个 physical key 完全一致。首次绑定与编译可能较慢，每条结果在采样、分析和校验完成后才打印。

本机没有连接该 VM，未执行真实 TPU 采样；表格数值只能来自你在 VM 上的实际运行，不内置论文数字。`AGENTS.md` 的路径用于默认设置，仍以 VM 实际文件和预检结果为准。
