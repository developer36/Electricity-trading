# 数据与市场规则

当前使用山东聚合数据。交易模拟还需核验价格单位、时段定义、申报截止、出清与结算口径，以及相关字段的可得时间。按规则版本记录适用日期。

## 2. 当前使用数据

### 2.1 山东电力市场价格与运行数据

来源：[ronaldowzy/price_forecast](https://github.com/ronaldowzy/price_forecast/tree/main/data) 的 `价格预测数据集.csv`。采用固定提交 `96ddcf9b020e98be35c9b8eb1c0f8b6e1fb8dc6f`，下载后检查大小和 Git blob SHA-1，再保存 SHA-256。无需 Zenodo 账号或访问申请。

| 项目 | 文件检查结果 |
| --- | --- |
| 范围 | 2024-01-01 至 2026-01-14，745 个交付日 |
| 规模 | 71,520 行、21 列；11,297,296 字节 |
| 网格 | 每日 96 行，时刻 `00:15` 至 `24:00` |
| 价格 | 日前价格、实时价格；保留空值与负价，不插值标签 |
| 供需预测 | 直调负荷、联络线受电、风电、光伏、非市场化核电、自备机组、地方电厂等 |
| 实际运行 | 对应实际值，以及抽蓄实际值等；使用时需核验发布滞后 |
| 日历 | 节假日、周末标记 |
| 天气 | 原始 CSV 没有天气列，由本项目另行补充 |

这是社区整理的聚合运行数据，尚未逐项对照官方原始披露核验。仓库[数据说明](https://github.com/ronaldowzy/price_forecast/blob/main/docs/data_privacy.md)提到了气象，但实际 CSV 不包含；以下载文件为准。代码许可与原始数据使用权限分别判断，保留来源与引用信息。

**价格粒度不能直接按表格行数认定。** 原始日前价格多数日期仅在整点有值，少数日期重复填充至 96 行，还有整日缺失。准备脚本保留原值，质量报告统计每天有效标签数；小时日前任务应单独确认时段定义、选取整点标签和聚合供需输入，不能前填或插值制造 15 分钟真值。实时价格也存在少量缺失。

本次全量检查：730 天各有24个日前标签，6 天各有96个填充值，9 天没有日前标签；实时价格非空71,432行。补充后合并表为 **71,520 行、34 列**，新增7个区域天气字段全部有值。天气覆盖17,904个小时、16个代表点，共286,464条城市小时记录。原始21列逐值保留，详细指标见本地 `data/metadata/quality_report.json`。

### 2.2 补充天气：山东16市代表点

使用 [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)，固定 `models=era5`，按月获取小时级数据。天气地点定义在 [configs/weather_locations.json](../../configs/weather_locations.json)，覆盖山东16个地级市的中心附近代表点，包含沿海和内陆地区。近似 WGS84 坐标会映射到 ERA5 网格，响应中的实际网格坐标保存在原始 JSON 中。

| 字段 | 单位 | 时刻含义 |
| --- | --- | --- |
| `temperature_2m` | °C | 整点温度 |
| `relative_humidity_2m` | % | 整点相对湿度 |
| `wind_speed_10m` / `wind_speed_100m` | m/s | 整点风速 |
| `cloud_cover` | % | 整点云量 |
| `precipitation` | mm | 前一小时累计降水 |
| `shortwave_radiation` | W/m² | 前一小时平均短波辐射 |

保留逐城市天气长表，并计算16点的等权均值作为初始区域代理。它不是全省面积、负荷或装机加权气象真值；后续可依据负荷分布和风光场站位置改进，不以单一省会天气代表全省。

**天气类型为事后再分析（`reanalysis_ex_post`），不是历史日前预报。** 合并表的天气列带 `_reanalysis_mean` 后缀，`weather_available_at` 留空，未知发布时间不伪造。当前可用于关联分析和事后解释；如用于历史输入，须另行建立发布延迟约束。不得将交付日实际天气直接作为日前预测的未来已知输入。要完成可交易的日前天气融合实验，后续需接入带起报时间和提前量的天气预报档案。

天气数据署名：Open-Meteo.com；ERA5：ECMWF/Copernicus。数据许可 CC BY 4.0；免费 API 有非商业用途及调用额度约束，使用范围见[官方条款](https://open-meteo.com/en/terms)。

### 2.3 下载与生成

使用 **Python 3.9+ 标准库**，无需第三方依赖。在仓库根目录运行：

```bash
# 下载山东数据，补充天气，生成对齐表与质量报告
python3 scripts/download_dataset.py

# 仅下载并校验山东电力原始文件
python3 scripts/download_dataset.py --prices-only

# 需要刷新所有原始文件时使用；会重新请求天气 API
python3 scripts/download_dataset.py --force

# 运行无需联网的逻辑测试
python3 -m unittest discover -s tests -v
```

默认不再请求广东 Zenodo 数据。重复运行校验并复用山东原文件；天气按请求参数生成缓存键，已经下载的月份通过结构、单位、时区、时间覆盖和缺失检查后复用。中断后重新运行即可继续尚未完成的月份。失败时返回非零退出码，不生成声称完整的合并表。每个输出文件写入临时文件，成功后原子替换。

```text
data/                                  # 整个目录被 Git 忽略
├── raw/
│   ├── shandong/prices.csv             # 固定版本原始电力数据
│   └── weather/era5_*.json             # 分月16地点天气响应
├── processed/shandong/
│   ├── weather_hourly_by_city.csv      # 逐城市小时天气
│   ├── weather_hourly_mean.csv         # 16代表点等权小时均值
│   └── prices_with_weather.csv         # 原21列 + 时间与天气字段
└── metadata/
    ├── price_source.json               # 提交号、URL、校验信息
    ├── weather_source.json             # 地点、单位、请求、许可、校验信息
    └── quality_report.json             # 覆盖、缺失、标签粒度和限制
```

`/data/`、`/outputs/`、`.env`、本地环境及缓存已由 [.gitignore](../../.gitignore) 忽略，不提交原始或处理后的数据，也不强制添加数据文件。

## 3. 时间对齐与特征约定

- 当前按源文件标签将时刻解释为**交付结束时间**：`2024-01-01 00:15` 对应 `00:00–00:15`，`24:00` 转为次日 `00:00`。该解释需在正式实验前与源方核验。
- `delivery_start` / `delivery_end` 使用带 `+08:00` 的北京时间；数据覆盖所有价格时段，天气额外获取至末日的次日，处理 `24:00` 边界。
- 对于15分钟区间，温度、湿度、云量和风速连接到区间起点所在整点；降水和辐射连接到其所在小时的结束标签。保留 `weather_instant_time` 和 `weather_interval_end`，不伪造15分钟气象精度。小时降水量重复关联到4行不表示每15分钟有这么多降水，不可将4行直接求和。
- 原始价格/供需字段原样保留，天气列为 `weather_<variable>_reanalysis_mean`。合并表用于研究审计，不能无差别将所有数字列送入模型。
- 特征按 `available_at <= forecast_origin` 选择。当前供需预测字段没有逐条发布时间，天气再分析也没有历史可得时间；需要明确假设或补齐元数据后才能宣称严格无前视回测。
- 季节性、节假日和历史价格作为基线；历史窗口候选为7/14/28天。目标时段的实际负荷、实际风光与实际天气不进入日前未来输入。
- 标准化、缺失填补和特征筛选仅在训练窗口拟合；标签缺失不填补后评分，负价和真实尖峰保留。

## 7. 参考与其他来源

- [PriceForecast 山东数据与说明](https://github.com/ronaldowzy/price_forecast)。
- [Open-Meteo 历史天气文档](https://open-meteo.com/en/docs/historical-weather-api)。
- 广东及其他暂未使用的数据来源：[数据集收集.md](数据集收集.md)。
