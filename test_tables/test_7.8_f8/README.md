# §7.8 Figure 8：只比较 Sectile

只需现有的 `~/test/sectile-work` 中的 **date_pk SF100 原表、26 个缓存数组、13 份精确参考结果**。不需要 `original` 目录，不下载或重新生成 SF100，不生成新的原表、缓存或参考结果。直接运行 Python 时只在终端打印 13 条查询的两列延迟、两列几何平均以及加速比；可选后台脚本会保存这些终端结果，便于断开 SSH 后回来查看。

将本文件夹全部内容上传到 `~/test_7.8_f8/`，使 `run_benchmark.py` 直接位于该目录，然后运行：

```bash
source ~/test/sectile-env/bin/activate
cd ~/test_7.8_f8
python -B run_benchmark.py
```

要关闭 SSH 浏览器后继续运行，改用以下命令；不要与已经启动的同一实验重复运行：

```bash
cd ~/test_7.8_f8
bash run_background.sh
```

后台脚本直接调用 `~/test/sectile-env/bin/python`，无需先激活环境，也不安装任何内容。仍可追加原入口参数，例如 `bash run_background.sh --queries Q1.1` 或 `--artifact-root ... --work ...`。它使用 `nohup`、重定向输入输出，并通过 `flock` 防止同一文件夹重复后台启动。`.benchmark.lock` 是零字节锁文件，不要在任务运行中删除它。

每次启动创建独立的小型 `results/run-时间戳-随机后缀/` 目录，保存 `result.txt`（原终端输出）、`pid`（后台包装进程 PID）和结束时生成的 `exitcode`。`results/latest` 指向最近一次后台任务，旧任务结果保留。不会保存新的 SF100 副本。重新连接 SSH 后查看：

```bash
cd ~/test_7.8_f8
cat results/latest/result.txt
# 实时查看；Ctrl+C 只退出查看，不停止实验
tail -f results/latest/result.txt
# 0 表示实验成功结束；其他数字表示失败
cat results/latest/exitcode
```

前置大表扫描、哈希与内存排序期间，`result.txt` 暂时为空是正常现象。`exitcode` 尚未生成可能是仍在运行，也可能是被强行终止；可用 `ps -fp "$(cat results/latest/pid)"` 检查包装进程。**浏览器/SSH 断开和 VM 停机是两件事**：`nohup` 能让任务在会话断开后继续，但无法在 VM 停机、重启或被抢占时继续；没有实现重启后的断点续跑。已打印的完整查询行会持续写入结果文件，便于之后查看已完成部分。

默认复用 `~/test/sectile-artifact-EF6C` 和 `~/test/sectile-work`。路径不同时：

```bash
python -B run_benchmark.py \
  --artifact-root "$HOME/test/sectile-artifact-EF6C" \
  --work "$HOME/test/sectile-work"
```

可先快速检查路径、内存和 TPU，检查不会扫描大表、重排或执行查询：

```bash
python -B run_benchmark.py --check-only
```

预检还会使用 4 字节匿名内存验证 memfd 加封、读取 seal 标志和只读映射，失败会在扫描大表之前停止。部分 Python 构建未暴露 `fcntl.F_ADD_SEALS` 等符号；代码在 Linux x86_64/aarch64 上按 [Linux UAPI](https://github.com/torvalds/linux/blob/master/include/uapi/linux/fcntl.h) 补全缺失的常量，仍实际要求内核应用并校验所有 seal，不会降级为可修改的共享索引。可单独快速自检（不加载 SF100、不初始化 TPU）：

```bash
python -B -c 'from source_order import check_sealing; check_sealing(); print("memfd sealing OK")'
```

实际运行会先完整读取一遍现有 `lineorder.tbl`，只解析 `LO_ORDERKEY`、`LO_LINENUMBER` 并校验整个文件 SHA。按这两个字段在 RAM 中建立排列索引，然后用同一个索引重排全部 9 个事实表缓存列。固定版本的 dbgen 按递增订单索引、订单内递增行号输出；程序检查无重复主键、连续订单/行号以及 SF100 的 150,000,000 个订单，从而恢复生成器的原始顺序，未使用随机打乱来代替。维表直接复用。日期排序路径直接读取原缓存；两种布局都使用各自实际内存列建立 zone map。

**准备阶段仍需时间**：读取已有大表、校验约 18 GB 缓存并排序约 6 亿行索引；它不计入查询延迟。该过程没有外部磁盘排序，也不会再占用七八十 GB 硬盘。索引约 2.24 GiB，保存在 Linux 匿名内存中，退出即释放；完整列重排和查询绑定还需要几十 GiB 主机 RAM。程序按实际缓存大小计算保守可用内存门槛（SF100 约 80 GiB，具体以预检为准），不足则提前停止。匿名内存仍受 Linux 交换策略影响；代码本身不会创建磁盘副本或临时排序文件。

每个查询、每种布局使用一个独立 TPU 进程，串行运行，两个完整输入目录不会同时驻留 HBM。使用已有固定版本环境，不升级依赖。运行前还会检查 v6e、单可见设备、SparseCore 和作者的 HBM 输入容量门槛。

复用作者 `production.run_resident_query`、成本模型、执行 guard、SQL 整理以及精确 INT64 校验；与 `reproduce.layouts` 相同的完整查询边界：一次预热、五次测量取中位数。包括在线 map build、join、聚合、同步、TPU 结果整理、最终 D2H 和主机 SQL 行构造；不包括加载、重排、H2D、规划、编译和正确性比较。Q4.3 使用作者 `paper_plan` 的完整逻辑计划。两种布局独立选择物理计划。

校验首先按作者方法认证现有 date_pk 参考结果的输入哈希、代码哈希、逻辑计划和参考负载。由于所有事实列共享已证明的一一行置换，精确 INT64 SQL 结果不变；在内存中显式记录参考继承关系和新布局的派生 dataset ID，再用原有逐次比较验证输出。不会伪造原始顺序的文件哈希，也不会把 date_pk 参考文件改写成另一套数据。另要求每条查询跨布局的有序 SQL 输出 digest 相同。作者源代码和原有 SF100 均保持不变。

两列几何平均分别为 `exp(mean(log(13 个中位延迟)))`，加速比为 `Source order / Date-sorted`，不保证等于论文的 2.87×。`--queries Q1.1` 可先测单条，此时汇总明确标为子集，但仍需要前置的大表扫描/内存索引。`--timeout` 默认每个查询/布局 3600 秒，不限制前置准备阶段。

正常运行屏蔽作者和底层库日志，只显示数据表；失败时显示错误和末尾诊断，不补造数值。作者要求写入的中间报告由适配器放在 RAM 中，Python 字节码和 JAX 持久编译缓存关闭；只有主动使用后台脚本时才额外保存上述小型结果和状态文件。适配器已进行本地小数据检查；完整 SF100/v6e 测量需在你的 VM 上完成。
