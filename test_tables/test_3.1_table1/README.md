# 论文 §3.1 Table 1 近似复现

在**已经配置好 JAX 的 GCP v6e TPU VM** 上，从本项目根目录执行：

```bash
python -B test_tables/test_3.1_table1/run_benchmark.py
```

脚本独立运行，无需安装本目录外的 Sectile 包。只使用第一个本地 TPU device（一个 v6e 芯片），其他可见芯片不参与。作者 artifact 的环境版本为 Python 3.11、`jax==0.10.2`、`jaxlib==0.10.2`、`libtpu==0.0.42.1`、`numpy==2.4.6`；这是源码提供的版本参考，本目录代码尚未在 TPU 上实测。环境必须提供 Pallas SparseCore API 和下述 XLA 编译选项；脚本不会安装依赖。

执行完毕后，终端输出三列表格：`Probes | JAX/XLA (ms) | Pallas/SC (ms)`，包含 128、2,048、5,466,180 三行。数字来自本次实测，不是论文数字的拷贝。

代码不下载数据、不保存 JSON/CSV、HLO、profile 或日志文件。原始耗时只存放在内存中，退出后释放；Python 字节码写入和 JAX 持久编译缓存也已关闭。

## 输入与计时

- 测量独立的 `supplier_map[keys]`，不是完整 JOIN 或 Q3.1。
- 当前目录没有原始 SSB 数据，因此使用固定种子的合成输入：`int32[2001]` 映射表、449 个非零供应商项、1–5 的组编号，剩余项为 0；probe keys 在 1–2000 中均匀生成，三档规模使用同一序列的前缀。它匹配原实验的规模和映射表选择率，但不还原原始数据值及 `date_pk` probe 顺序。
- SparseCore 内核保留作者 [`build_sparsecore_gather`](../../sectile-artifact-EF6C/reproduce/protocols/probe.py) 的实现：HBM→VMEM 表复制、流水线、gather、padding 和输出裁剪。
- 每条路径默认预热 5 次、计时 20 次，交替执行顺序，取中位数。输入提前放到 TPU；计时包围一次已编译调用和 `block_until_ready()`，包含主机调用及同步开销，以及 SparseCore 每次调用的片上 staging、padding 和流水线开销。编译、预热、初始主机传输、结果回传和 NumPy 逐元素校验不计时。
- 新版 XLA 可能自动将普通 gather 放到 SparseCore。为保留论文 TensorCore 与 SparseCore 的比较，JAX/XLA 路径显式将 `xla_tpu_offload_gather_to_sparsecore` 和 `xla_tpu_offload_all_supported_gathers_to_sparsecore` 设为 `false`（[JAX 官方说明](https://github.com/jax-ml/jax/issues/40553)）。不支持这些选项或 SparseCore 的环境会报错。

可选参数：`--warmups 5 --samples 20 --seed 0`；采样次数必须是至少 2 的偶数。输入顺序、JAX/XLA 版本和运行时开销会影响结果，不保证与论文毫秒数或 65.5× 加速比一致。
