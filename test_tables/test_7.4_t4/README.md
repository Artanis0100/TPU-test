# §7.4 Table 4：固定流水线的 join-map 表示

复用 VM 现有 SF100 数据与环境，输出 Q2.3、Q3.3 的四列 latency（ms）：Compiler-managed、All-Hash、All-Bitmap、All-Bitmap-Rank。正常运行只打印表格；内部日志静音，不生成结果、日志或 profile 文件，不下载或安装任何内容。

把本文件夹的**全部内容**解压到 `~/test_7.4_t4/`，使该目录直接包含 `run_benchmark.py` 等文件，然后执行：

```bash
source ~/test/sectile-env/bin/activate
cd ~/test_7.4_t4
python -B run_benchmark.py --check-only
python -B run_benchmark.py
```

默认作者源码为 `~/test/sectile-artifact-EF6C`，已有工作目录为 `~/test/sectile-work`。需要时用 `--artifact-root`、`--work` 指定其他已有路径。可先测试单条查询或部分表示：

```bash
python -B run_benchmark.py --queries Q2.3
python -B run_benchmark.py --queries Q3.3 --variants bitmap bitmap_rank
```

每个“查询 × 配置”使用独立子进程，始终串行执行。默认每个配置最多等待 3600 秒，可用 `--timeout 7200` 调整。Compiler-managed 每次完整查询可能需要几十秒；编译与测量期间会保持静默，请勿同时运行其他 TPU 任务。

实验直接复用作者 `reproduce.compiler_only.run_cell`、`reproduce/protocols/table3.py` 与 `evaluation/table3/table3_q23_hash_chunked_formal.py`。作者内部文件名 table3 对应本论文 **Table 4**，不代表运行了 §7.3 的硬件计数表。

三种 custom 表示保留同一查询的 join 顺序、stage 边界、中间物化和聚合实现，仅强制所有非 DATE join map 使用指定表示。Q2.3 Hash 使用作者已有的 Part 一块、Supplier 两块恢复方案，并保留原始编译 stage-signature、DATE 不变性和无新增 stage/物化的检查。每个 map 的分块数量仍由作者固定设计决定，不将这里的强制变体加入生产规划器。

四组均为 1 次预热、5 次完整 SQL-ready 测量取中位数。计时包含 online map build、join、聚合、守卫/必要重试、TPU 结果整理、D2H 和主机 SQL 行解码；规划、编译、初始加载和正确性比较不计时。原始精确聚合、有序 DuckDB SQL 行和无溢出校验全部保留；失败或无效测量会报错，不填 0 或论文数字。

作者脚本需要先写证据再读取校验，因此这里将这些输出保存在作用域内的 RAM 路径中，保留 JSON/NPZ 的真实字节、统计散列、glob 与读回检查；退出即释放。只接管已知模块的输出 Path 别名与输出序列化函数，原始源码、缓存和参考仍按真实路径读取，不修改作者文件。Python 字节码和 JAX 持久缓存关闭；Python/C++ 日志在匿名内存描述符中捕获，普通异常显示诊断尾部。

`--check-only` 不执行完整 SF100 查询。已做本地语法、RAM 输出读回和作者协议对接检查，未连接你的 VM 做真实 TPU 测量；预检通过仍需实际运行验证。
