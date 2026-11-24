# §7.4 Figure 6：Compiler-managed only 与 Full library

复用 VM 已有的 SF100 数据、精确参考和 Sectile 环境，串行测量全部 13 条 SSB 查询的两种配置。正常完成时，终端只显示表头和 13 行结果，每行两项中位 latency，单位 ms。不绘图、不保存结果或日志文件、不重新生成数据、不下载或安装依赖。

## 上传后运行

把本文件夹的**全部内容**解压到你指定的 `~/test_7.3_f6/`，确保该目录直接包含 `run_benchmark.py` 及其他 Python 文件。本地文件夹名为 `test_7.4_f6`，不影响 VM 目录名。

```bash
source ~/test/sectile-env/bin/activate
cd ~/test_7.3_f6
python -B run_benchmark.py --check-only
python -B run_benchmark.py
```

默认复用以下已有路径，无需复制 SF100 数据：

- 作者源码：`~/test/sectile-artifact-EF6C`
- 数据、整数缓存与参考结果：`~/test/sectile-work`

若 VM 路径不同，显式指定：

```bash
python -B run_benchmark.py \
  --artifact-root "$HOME/test/sectile-artifact-EF6C" \
  --work "$HOME/test/sectile-work"
```

可以先只测试一条，或指定多条：

```bash
python -B run_benchmark.py --queries Q1.1
python -B run_benchmark.py --queries Q2.1 Q3.1
```

按查询组运行的入口为 `run_benchmarkQ1.py`、`run_benchmarkQ2.py`、`run_benchmarkQ3.py` 和 `run_benchmarkQ4.py`，用法与总入口一致，例如：

```bash
python -B run_benchmarkQ2.py
```

`--check-only` 检查现有路径、参考与缓存，以及独立子进程中的作者源码完整性、固定依赖和单芯片 v6e 环境；不加载完整 SF100 列、不运行完整查询。预检通过不能代替实际查询的正确性验证。

每个“查询 × 配置”默认最多等待 3600 秒，可用 `--timeout 7200` 调整。作者 README 给 compiler-only 全组的参考预算约 45 分钟，包含加载和首次编译；实际耗时取决于 VM。程序每完成一条查询的两组测量才打印一行，期间静默是正常现象。请勿同时运行其他 TPU 实验。

## 测量内容

- **Compiler-managed only**：调用作者 `reproduce.compiler_only.run_cell`，保留强制 `compiler_managed_only` 候选、`exact_int64`、`hardware_mode='vpu'` 和 `compiler_scan_admission.admit_compiler_range`。这组由 XLA 决定高层操作的硬件映射，不额外添加 SparseCore offload 开关。
- **Full library**：调用作者 `reproduce.main_matrix.run_cell`，采用 Figure 4a 的自动选择配置，允许自定义 Pallas 算子和融合流水线。Q4.3 保留作者的 `paper_plan`。
- 两组均执行完整逻辑查询，包括 Date join；保留 zone-map guard、原有范围准入和精确 fallback。每个配置每条查询使用新的独立子进程，始终串行执行，避免前一组的 TPU 分配残留。
- 两组均为 **1 次不计时预热 + 5 次测量，取中位数**。时间从缓存 executable 接收 TPU 常驻输入开始，到完整 SQL 结果完成 D2H、主机解码并按要求排序为止。规划、编译、初始数据加载和正确性比较不计时。
- 每次结果仍须通过作者精确聚合与有序 SQL 行比较；失败、超时或结果无效时非零退出，不用 0 或论文数字补齐。

论文 Figure 6 的 Full library 数值复用了 Figure 4；本程序重新测量两组，输出的是本次 VM 的实测值。两组扫描范围按作者各自计划保留，不人为改成相同范围。

## 静默与不落盘

适配器仅在运行时接管作者的结果打印和输出文件写入，原始计算、计时、守卫与验证函数保持不变。输入、源码和参考文件仍正常只读访问。Python 字节码和 JAX 持久编译缓存关闭，不生成 CSV、JSON、profile 或报告。

成功运行的子进程 stdout/stderr 在原生文件描述符 1/2 层捕获到 Linux 匿名内存描述符中，覆盖启动、测量和退出阶段，因此 Python、JAX、libtpu 和其他 C++ 内部日志不会刷屏，也不会保存到磁盘。普通异常会显示错误与末尾诊断信息，便于定位问题；若原生进程直接崩溃，则报告退出状态。已经成功测量的行不会改写。程序只复用现有作者源码，不修改源码、SF100 数据或依赖版本。

本地未连接你的 TPU VM，未执行真实 SF100 TPU 测量。预检或运行失败时保留终端错误，先不要升级现有环境。
