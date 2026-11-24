# §7.2 Figure 4(a)：Sectile，SF100

测量 Figure 4(a) 中 Sectile 的 13 个 SSB 查询，使用真实 SF100 数据、`date_pk` 布局和一颗 GCP TPU v6e。复用完整作者 artifact 的自动物理规划器、论文参数和算子实现；不修改作者源码。

保留 §7.1 的完整查询计时口径：`exact_int64`、TPU 结果整理、zone-map 检查及必要的精确 fallback；每个查询预热 1 次、测量 5 次，打印中位数。计时结束于主机拿到解码、排序后的 SQL 行，不是仅测 TPU kernel。加载、规划、编译和参考结果生成在计时外，精确聚合及 SQL 行校验继续执行。

## 1. 上传目录和准备环境

需要上传完整的 `sectile-artifact-EF6C` 和本文件夹，保留以下相对位置，不能只上传单个 Python 文件：

```text
Sectile/
  sectile-artifact-EF6C/
  test_tables/test_7.2_4a/
  sectile-work/                 # 数据准备时创建
```

在已有 Python 3.11 的 GCP TPU VM Bash 中执行；数据生成还需要 Bash、CMake、C 编译器、make、GNU coreutils、awk 和 findutils。生成器源码已附带，本地准备入口不需要 Git 或在线下载生成器。作者固定版本包括 JAX/JAXLIB `0.10.2`、libtpu `0.0.42.1` 和 DuckDB `1.4.4`；完整依赖以作者文件为准：

```bash
cd /path/to/Sectile/sectile-artifact-EF6C
python3.11 -m venv ../sectile-env
source ../sectile-env/bin/activate
python -m pip install -r requirements-tpu.txt
export PYTHONPATH="$PWD/src:$PWD"
export PYTHONDONTWRITEBYTECODE=1
```

环境必须只向 JAX 暴露一颗 v6e。作者代码会检查固定依赖版本、artifact 内容清单和论文参数哈希；修改原 artifact 会导致检查失败。

## 2. 准备真实数据与参考结果

`generator/ssb-dbgen/` 已附带作者使用的 [SSB 生成器源码](https://github.com/eyalroz/ssb-dbgen/tree/ae1e254aa4d603d8ef1f44078e5abed011634b23)，固定 commit 为 `ae1e254aa4d603d8ef1f44078e5abed011634b23`。`ssb-dbgen-source.json` 记录全部 33 个文件的原 Git blob、大小和 SHA-256。附带的是源码，尚未生成或下载 SF100 数据。

在 GCP 上用本目录的新入口生成真实 SF100 的 `date_pk` 数据、整数缓存，以及全部 13 个查询的精确参考结果：

```bash
cd ..
python -B test_tables/test_7.2_4a/prepare_data.py --with-references
```

不带 `--with-references` 时只准备表和整数缓存；已有表和缓存会通过作者的检查后复用。`--work`、`--artifact-root` 与 benchmark 使用相同默认路径，也可显式指定：

```bash
python -B test_tables/test_7.2_4a/prepare_data.py --work /data/sectile-work --artifact-root /path/to/Sectile/sectile-artifact-EF6C --with-references
```

入口先校验全部源码，再复制到 `sectile-work/ssb-dbgen/` 本地编译；已有源文件不匹配时直接报错，不覆盖。`generator/prepare_frozen_ssb_data.sh` 和 `generator/prepare_data.original.py` 是作者文件的原样副本。`prepare_local_ssb.sh` 保留原脚本的生成、排序、manifest 和恢复流程，只把 Git 下载步骤替换为本地 CMake 编译。整数缓存仍由作者原 loader 构建，参考结果仍由作者 `reproduce.references` 生成，原 artifact 不作修改。

已经准备好的输入可直接复用。所需目录为：

```text
sectile-work/data/date_pk/sf100/              # 五张 .tbl 表和 manifest.txt
sectile-work/cache/date_pk/sf100-v1/          # manifest.json 和整数 .npy 列
sectile-work/references/date_pk/sf100/Q11/    # 每个查询各有一个参考目录
...
sectile-work/references/date_pk/sf100/Q43/
```

每个参考目录包含 `result.json`、`aggregates.npz`、`sql-rows.json`。缓存记录源目录和文件身份，移动数据或修改文件后不能直接假定旧缓存仍有效。

SF100 排序和 CPU 参考生成可能需要数小时。作者对**整个研究**给出的资源建议约为 350 GB 可用磁盘、170 GB 主机内存；这不是本目录一次 SF100 测量的精确资源需求。数据准备会占用持久存储，benchmark 阶段只输出终端结果。

## 3. 运行 benchmark

继续使用上述环境，从 Sectile 目录执行：

```bash
python -B test_tables/test_7.2_4a/run_benchmark.py
```

也可按查询族分别运行；这些入口使用相同的算法和计时口径：

| 入口 | 查询 |
| --- | --- |
| `run_benchmark.py` | 全部 13 个查询 |
| `run_benchmarkQ1.py` | Q1.1、Q1.2、Q1.3 |
| `run_benchmarkQ2.py` | Q2.1、Q2.2、Q2.3 |
| `run_benchmarkQ3.py` | Q3.1、Q3.2、Q3.3、Q3.4 |
| `run_benchmarkQ4.py` | Q4.1、Q4.2、Q4.3 |

例如：

```bash
python -B test_tables/test_7.2_4a/run_benchmarkQ1.py
python -B test_tables/test_7.2_4a/run_benchmarkQ4.py --work /data/sectile-work --artifact-root /path/to/Sectile/sectile-artifact-EF6C
```

`--work` 默认指向 `Sectile/sectile-work`，`--artifact-root` 默认指向 `Sectile/sectile-artifact-EF6C`，均不依赖执行命令时的当前目录。请串行运行，避免多进程争用 TPU。

benchmark 为每条查询启动独立进程，完成后打印一行中位延迟，全部运行会得到 13 行数字，不画图。耗时和记录只保留在内存中，不下载数据、不生成结果 JSON、CSV、HLO、profile 或实验日志。`console_artifacts.py` 在运行时拦截作者 harness 对其输出路径的报告写入；原输入读取、规划、计算、计时和正确性检查仍保留。数据或参考缺失时会报错，不自动准备输入。Q4.3 使用作者的 `paper_plan`，包含官方 SSB 的 CUSTOMER join 和 `C_REGION = 'AMERICA'`。

当前尚未在真实 v6e 上完成这套 SF100 基准验证，运行后表格中的数值才是当前硬件和软件环境的实测结果。
