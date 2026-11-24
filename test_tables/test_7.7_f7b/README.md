# §7.7 Figure 7b：Join plus projection

复用作者的 Q2.1-derived SUPPLIER join-plus-projection 微基准，输出 1、2、4、6、8、16、24、32 个 projection terms 时的三列 latency（ms）：JAX/XLA、Pallas SparseCore、Pallas MXU。正常运行只打印结果表格，不绘图、不保存实验记录或日志、不修改原作者源码。

## 上传后运行

将本文件夹的全部内容解压到 `~/test_7.7_f7b/`，确保该目录直接包含 `run_benchmark.py` 和其他 Python 文件。

```bash
source ~/test/sectile-env/bin/activate
cd ~/test_7.7_f7b
python -B run_benchmark.py --check-only
python -B run_benchmark.py
```

默认读取作者源码 `~/test/sectile-artifact-EF6C`，并依次查找 **SF1** 原始表目录 `~/test/sectile-work/data/date_pk/sf1` 和 `~/test/sectile-work/figure7-data/sf1`。该目录至少需要 `lineorder.tbl`、`part.tbl`、`supplier.tbl`。本图使用论文规定的 SF1，现有 SF100 不能直接代替；基准不会自动下载、生成数据或截取 SF100 冒充 SF1。若 SF1 位于其他目录，显式指定：

```bash
python -B run_benchmark.py --data-dir /path/to/real/sf1
```

如果目前只有 SF100，从两个 Figure 7 文件夹中的任意一个显式执行一次：

```bash
python -B prepare_sf1.py
```

它复用 VM 已有的 `~/test/sectile-work/ssb-dbgen/build/dbgen` 和作者原始生成、date_pk 排序、验证流程，离线生成新的 `~/test/sectile-work/figure7-data/sf1`，供两个基准共用。只有此准备命令会写 SF1 输入及数据 manifest；不重新生成 SF100，不制作缓存、查询参考或实验记录。目标父目录必须全新，已有 SF1 时跳过此命令。自定义路径用 `--dbgen-dir`、`--data-root`，随后通过基准的 `--data-dir 新目录/sf1` 指定。

可先做一项小检查，完整运行时省略 `--widths`：

```bash
python -B run_benchmark.py --widths 1
```

首次准备 route、编译和测量期间会保持静默，整个扫描默认超时为 3600 秒，可用 `--timeout` 调整。请与其他 TPU 实验串行运行；`--check-only` 检查现有输入和环境，不代表已经执行完整数据校验或真实查询测量。

## 测量与作者源码

直接执行 `ablation_studies/mxu/projection.py` 的原始 `main()`，参数与 `reproduce.mxu` 一致：SF1、3 次预热、30 次测量取中位数、1024 行固定 tile。八个宽度在一个独立 TPU 子进程内完成，数据及持久 route 只准备一次。

计时使用作者 `online_supplier_join_plus_projection` 的完整 online `median_ms`：包含设备端目录选择、predicate payload binding、SUPPLIER equality join 和 projection。不会改用只测 prepared kernel 的 `prepared_median_ms`。表加载、持久 route 构造、初始 H2D、编译、结果 D2H 和正确性比较在计时外。

三组使用相同 route、join key 和 payload。SparseCore 保留作者的 8-lane 分块及单个外层 JIT 编排；MXU 保留物理宽度向 8 对齐及 padding 检查。所有 projection 输出都执行原始精确数组比较；失败即报错，不补零或内置论文数字。

这是算子微基准，不是完整 Q2.1 latency。论文 Figure 7b 的 SF1 约有 46,026 条选中行；实际行数由输入及作者原始 predicate/route 计算，本程序不强制制造该数字。

## 静默与不落盘

作者的 CSV、summary 和 manifest 仅保存在局部 RAM 输出路径中，源码及输入读取保持不变。Python 字节码、JAX 持久编译缓存关闭；正常 Python/C++ 日志在匿名内存中捕获。普通错误显示末尾诊断并非零退出。

已做本地接口和模拟验证；尚未连接你的 VM 执行真实 TPU 测量。环境错误时先保留报错，不升级现有依赖。
