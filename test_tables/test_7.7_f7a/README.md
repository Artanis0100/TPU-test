# §7.7 Figure 7a：Aggregation 宽度扫描

直接复用作者 `ablation_studies/mxu/aggregation.py`，比较 JAX/XLA、Pallas SparseCore、Pallas MXU。终端打印 17 个 aggregate-term 数量各自对应的三项 median latency（ms），不绘图、不保存结果或日志文件。

把本文件夹全部内容解压到 `~/test_7.7_f7a/`，使入口直接位于该目录：

```bash
source ~/test/sectile-env/bin/activate
cd ~/test_7.7_f7a
python -B run_benchmark.py --check-only
python -B run_benchmark.py
```

默认作者源码为 `~/test/sectile-artifact-EF6C`。**本实验使用真实 SF1 原始表，不使用 SF100 缓存**；依次查找 `~/test/sectile-work/data/date_pk/sf1` 和 `~/test/sectile-work/figure7-data/sf1`。需要该目录中的 `lineorder.tbl`、`customer.tbl`、`supplier.tbl`、`date.tbl`。路径不同时用 `--data-dir /已有/SF1目录` 指定。基准不会自动下载、生成、截断或覆盖数据。

如果目前只有 SF100，先从两个 Figure 7 文件夹中的任意一个显式执行一次：

```bash
python -B prepare_sf1.py
```

这一步复用 VM 已有的 `~/test/sectile-work/ssb-dbgen/build/dbgen`，调用作者原始生成、date_pk 排序和验证流程，离线写入新的 `~/test/sectile-work/figure7-data/sf1`。只生成 SF1 输入表及数据 manifest，不重新生成 SF100、不制作查询缓存或参考结果。已有目标父目录会被拒绝，避免覆盖；已有 SF1 时跳过此命令。两个基准随后会自动发现并共用这份 SF1。自定义路径可用 `prepare_sf1.py --work ... --dbgen-dir ... --data-root 全新目录`，再给基准传 `--data-dir 全新目录/sf1`。

可先运行 `python -B run_benchmark.py --widths 1`；完整运行时省略 `--widths`。`--timeout` 默认 3600 秒，适用于整个扫描。加载、编译和测量期间保持静默，所有宽度校验完成后打印表格；请串行运行两个基准。

扫描宽度为 `1,5,10,15,20,25,26,27,28,30,32,33,34,35,36,40,48`，显式补全图中 32 以后的宽度；作者源码的默认列表只到 32。

原始准备阶段在主机执行 Q3.1 的过滤与 join，生成 surviving group codes 和各项 SQL-like measure，补齐到 262,144 行，再传到 TPU。这里只计 aggregation 算子，**不是完整 Q3.1 latency**。算术使用作者原始 float32；输出与独立 float64 主机 reference 以 `rtol=5e-4, atol=1.0` 检查。

每种实现先执行一次首次编译调用，再做 3 次预热和 30 次计时调用，返回 30 次的中位数。计时含 warm TPU dispatch、同步和 aggregation 执行；不含加载、主机过滤/join、measure 构造、初始 H2D、编译预热、最终 D2H 和正确性校验。SparseCore 超过单次 accumulator 容量时沿用作者分批 measure pass，所有 pass 保留在同一个 outer JIT 调用内；MXU 保留原有 8-lane 对齐和 16,384-row tile。

只把作者用于输出 CSV、summary、manifest 的模块内 Path 别名接到内存；真实输入读取保持不变，计算、计时和检查均执行原作者函数。Python/C++ 日志由入口在匿名内存描述符中捕获，正常完成只打印表格，异常显示诊断。Python 字节码与 JAX 持久缓存关闭，原作者源码和依赖版本不修改。

本地未连接 TPU VM，尚未执行真实 TPU 测量。预检失败时保留错误，先不要升级现有环境。
