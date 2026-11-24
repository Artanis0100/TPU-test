# §7.5 Table 5：Q2.3 计划预测与实测

复用 VM 已有的 SF100 数据、精确参考、作者源码和依赖环境。终端只打印 Bitmap、Bitmap-Rank、Hash 三行的 predicted time、measured time（ms）及预测/实测排名；`*` 来自规划器实际选择。不内置论文数字，不下载或生成数据，不保存结果、日志或 profile 文件。

## 上传后运行

把本文件夹的全部内容上传并解压到 `~/test_7.5_t5/`，让 `run_benchmark.py` 直接位于该目录中：

```bash
source ~/test/sectile-env/bin/activate
cd ~/test_7.5_t5
python -B run_benchmark.py --check-only
python -B run_benchmark.py
```

默认读取 `~/test/sectile-artifact-EF6C` 和 `~/test/sectile-work`。路径不同时显式传入：

```bash
python -B run_benchmark.py \
  --artifact-root "$HOME/test/sectile-artifact-EF6C" \
  --work "$HOME/test/sectile-work"
```

三组始终串行，每个候选使用新的子进程。默认每组最多等待 3600 秒，可用 `--timeout 7200` 调整；加载、规划、预检和首次编译期间保持静默，请勿同时运行其他 TPU 实验。`--check-only` 检查路径、原始源码与固定依赖、单芯片 v6e 环境，不加载完整 SF100 列或测量查询。

## 与原作者协议的对应

作者将本文 Table 5 的入口命名为 `reproduce/table4.py`，其实际实现是 `reproduce/protocols/table4.py`。本程序直接执行该 formal protocol，保留 `q23_date_invariant_preselection`、固定参数文件、原始统计、DATE map/stage 一致性检查和所有绑定、精确验证。候选名字指 **PART map** 的表示；三组的 SUPPLIER map 都为 Bitmap，DATE 保持相同的 terminal join。这里不强制所有 map 都使用同一种表示；那是 Table 4 的另一项实验。

预测值来自冻结模型，三种候选在任何实测反馈之前确定。测量执行作者的 **1 次预热 + 5 次测量，取中位数**，使用 `measure_numeric_calls` 的数值结果就绪区间 A：包括常驻输入上的在线 map 构建、join、聚合、原有调度和同步；在 `jax.block_until_ready()` 完成时停表。计时不含最后完整结果 D2H、TPU SQL 整理、主机解码和正确性比较，因此不能与 Figure 4a、Figure 6 或 Table 4 的完整 SQL 结果 latency 直接等同。

原作者完整 SQL 与 numeric 输出预检仍在计时前执行，每次测量后的精确聚合检查也保留。程序核对三组的源码、数据、参数、统计、预选计划及 DATE 合约身份，并要求精确聚合和有序 SQL 行逐组相同。排名由本次输出的预测值和实测中位数分别计算；星号仅来自原始 `planner_selected`，不根据最快实测值推断。若当前规划器的选择不在这三组中，会报错，不伪造星号。

## 静默与内存输出

输入、源码、参数和参考文件保持真实只读访问。作者需要写后读、哈希核对的统计、NPZ、JSON 和 HLO 输出通过局部 Path/写入接口进入进程内存，原始统计序列化和哈希校验继续执行；不会创建输出目录。Python 字节码和 JAX 持久编译缓存关闭。正常 Python/C++ 日志在匿名内存描述符中捕获，仅显示最终表格；普通异常显示错误及末尾诊断后非零退出。

本地未连接 TPU VM，尚未执行真实 SF100 TPU 测量。预检或测量失败时保留报错，先不要升级现有依赖或重新生成数据。
