# §7.6 Table 6：容量低估与精确 fallback

复用已有 SF100 数据、精确参考、作者源码和 TPU 环境。正常运行只打印 Q1.1、Q2.1、Q3.1 的 Baseline、−10%、−30%、−50% 四列完整查询 latency（ms）。`†` 根据本次五次正式测量中实际返回的容量 fallback 标志添加，不内置论文数字或预设哪些格子必须触发。

把本文件夹的**全部内容**上传并解压到 `~/test_7.6_t6/`，确保 `run_benchmark.py` 直接位于其中，然后执行：

```bash
source ~/test/sectile-env/bin/activate
cd ~/test_7.6_t6
python -B run_benchmark.py --check-only
python -B run_benchmark.py
```

默认读取 `~/test/sectile-artifact-EF6C` 和 `~/test/sectile-work`，无需复制或重新生成 SF100。路径不同时使用 `--artifact-root`、`--work` 指定已有目录。

可以先测一条查询或少量缩减设置；Baseline 始终包含在输出中：

```bash
python -B run_benchmark.py --queries Q1.1
python -B run_benchmark.py --queries Q1.1 --reductions 0.5
```

每个“查询 × 容量设置”使用独立子进程，始终串行执行。默认每组最多等待 3600 秒，可用 `--timeout 7200` 调整。精确 fallback 可能使单次查询变成数十秒，加载和首次编译也需要时间；正常运行期间保持静默，每完成一条查询的全部设置后才输出一行。请勿同时运行其他 TPU 任务。

## 与作者协议的对应

直接调用 `reproduce.misestimation.run_cell`：先使用原始统计与冻结模型正常选择计划，再调用作者 `misestimation_capacity.inject` 和 `lowering_injection`，仅改变估计导出的中间缓冲容量。保留 join 顺序、map、分块、算术、谓词、已有 overflow 检测和精确 fallback，不重新排名或拟合模型。不同容量格子的正常候选顺序、选择和预测必须保持一致。

容量按作者原有合法对齐粒度向上取整，`−10%` 是请求减少量，不保证物理容量恰好减少 10%；论文数据中的这一设置取整后不变，本程序保留实际分配结果，不强制制造相同结果。

每组 **1 次预热 + 5 次正式测量，取中位数**。采用完整 SQL-ready 口径，包含 online map build、join、聚合、原有精确 fallback、同步、TPU 结果整理、最终 D2H 和主机 SQL 行构造。初始数据加载、规划、编译和每次计时后的精确聚合/有序 SQL 比较不计时。沿用作者父入口的 `OMP_NUM_THREADS=8`、`OPENBLAS_NUM_THREADS=8`。

所有六次调用都必须通过精确验证。此表允许目标容量不足导致的精确分支；join-map overflow、scan-range 外部重试、失败或无效测量仍会报错，不用 0 或论文数字补齐。`†` 只依据五次正式测量的 `capacity_exact_branch_calls`，不依据预热，也不把 overflowing page 数量误当成多次重试。

## 静默与不落盘

作者生成的 JSON、NPZ、HLO 和审计记录保存在作用域内 RAM 路径中，保留原始序列化和读回校验，退出即释放。输入、源码、参数与参考仍按真实路径只读访问，不修改原作者文件。Python 字节码、JAX 持久缓存关闭，不生成结果、日志、profile 或图表文件。

正常 Python/C++ stdout/stderr 日志在匿名内存描述符中捕获；普通异常显示错误和末尾诊断并非零退出。`--check-only` 检查路径、固定依赖、源码完整性和单芯片 v6e 环境，不加载完整 SF100 列或执行完整查询。

已完成本地协议与模拟检查，尚未在真实 TPU 上测量。预检失败时保留终端错误，先不要升级现有环境或重新生成数据。
