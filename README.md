比赛背景：
实现司机找货智能体（Agent）：系统在每轮交互中仅提供必要的司机身份类信息，智能体须自主获取环境状态与可选行动，在模拟的一个自然月内持续推进决策，在遵守业务约束的前提下，实现月度净收益的提升，并尽可能满足与司机运营相关的个性化偏好。
要求：
- 信息获取方式：除每轮给定的最小必要输入外，其余状态与货源信息须通过环境与赛方规定的交互方式取得。
- 仿真时长：在一次评测中，智能体须在完整月度周期内参与循环决策。
- 动作空间：在任一决策时刻，智能体须在接单、休息与空驶三类动作中择一执行。
- 优化目标：以月度尺度上的综合表现为评价对象，兼顾净收益与个性化偏好，而非孤立地追求单笔订单收益。
约束：
参赛 Agent / 决策代码在仿真运行中：
- 禁止直接读取或解析 demo/server/data/cargo_dataset.jsonl、demo/server/data/drivers.json（及同内容文件的任意磁盘路径），包括将路径写死、open 整表扫描、按行缓存全库等。
- 必须仅通过评测环境提供的接口获取决策所需信息，例如：SimulationApiPort.query_cargo（货源候选）、get_driver_status（司机状态，含可见窗口内的 preferences）、query_decision_history（本会话已执行步骤）等。
- 单个司机不超过500万token，总仿真运行失常不超过4个小时
数据说明：
货源数据：
每条货源常用字段：
- cargo_id：唯一标识
- create_time / remove_time：货源有效时间区间（墙钟）
- start / end：起终点坐标（lat/lng）
- load_time：装货时间窗（可为空）
- cost_time_minutes：运输耗时（分钟，已包含装卸货时间）
- price：货源价格（原始数据单位为分）
评分注意：
- 收益计算以 cargo_id 回查原始货源价格，不以动作结果中价格字段为准。
- 收益脚本按 price / 100 换算为元后参与 gross_income 计算；结果文件中的金额口径均为元。
- 接单合法性会用到 create/remove/load_time 等时间字段。
司机数据：
常用字段：
- driver_id：司机 ID
- current_lat / current_lng：初始位置
- cost_per_km：单位里程成本
- preferences：个性化偏好（可为空）
偏好说明：
- get_driver_status 返回的每条偏好仅含 content、penalty_amount、penalty_cap；数据文件中的 start_time / end_time 为可见窗口（控制何时向选手展示该条），不通过接口返回。
- 当前为“文本偏好 + 固定规则映射”，不是通用规则语言。
- 选手侧请勿在决策代码中硬编码偏好规则（例如为某 driver_id 写死 if/else、固定时间窗等）。应基于运行时接口返回的 preferences 等状态，由策略或模型理解并执行；硬编码不仅难以覆盖文本变更，也与评测“读数据、做决策”的预期不符。
- 官方收益脚本为赛后核对服务；偏好文本若调整，主办方会同步维护 calc_monthly_income.py 中的规则实现，与选手代码无关。
动作日志：
每步核心字段：
- 基本信息：step、driver_id
- 时间：step_elapsed_minutes、query_scan_cost_minutes、action_exec_cost_minutes
- 位置：position_before、position_after
- 决策：action.action、action.params
- 结果：result（含 simulation_progress_minutes、simulation_wall_time 等）
用途：
- 该日志是收益脚本动作合法性校验的直接输入。
- 校验按司机隔离，某个司机失败不会影响其他司机。
时间与单位口径：
- 仿真主时间单位：分钟
- simulation_progress_minutes 起点：2026-03-01 00:00:00
- 墙钟时间用于展示与解释，校验逻辑以分钟为准
仿真耗时规则（距离 → 时间）
以下规则与 simkit 中实现一致；日志里的 query_scan_cost_minutes、action_exec_cost_minutes 均基于同一套推进逻辑。
距离：
- 任意两点间里程为 Haversine 大圆距离，单位 km；界面与日志中的距离多为保留两位小数的浮点数。
- 涉及时间的换算时，先将距离折算为分钟，再对分钟 ceil 向上取整
空驶速度与时间（reposition）：
- 速度：取评测配置中的 reposition_speed_km_per_hour（单位 km/h，见 demo/server/config/config.json）。
- 耗时（分钟）：设当前边的里程为 d km（d > 0），速度为 speed km/h，则
minutes = max(1, ceil(d / speed * 60)) —— 先按 (d / speed) * 60 换算成分值，再 ceil 向上取整为整数分钟，且不少于 1 分钟。
- 零距离：若目标与当前位置重合（d = 0），仍按引擎规则计最少 1 分钟（与「接单时空驶」的零距离口径不同）
查看货源：
- 不改变司机位置，仅查询指定经纬度附近的候选货源列表。
- 返回条数：参数 k（默认 100）；超出 1..600 时自动截断到该区间。按与查询点 Haversine 距离升序取当前 online 池中最近的至多 k 条。
- 仿真时间消耗：设本次实际返回条数为 n，浏览批大小为 cargo_view_batch_size（Demo 默认 10），则
scan_minutes = ceil(n / cargo_view_batch_size)；若 n = 0 则为 0 分钟。
- 该耗时计入当步的 query_scan_cost_minutes。
接单（take_order）：空驶 → 装货窗等待 → 干线：
接单全程使用与空驶 相同的 reposition_speed_km_per_hour 计算「开到装货点」的行驶时间。
1. 空驶到装货点 start  
  - 里程：当前位置到货源 start 的 Haversine 距离，记为 d_pickup km。  
  - 若 d_pickup <= 1e-6 km（视为已在装货点）：空驶耗时 0 分钟，仅同步位置到 start。  
  - 若 d_pickup > 1e-6 km：minutes = max(1, ceil(d_pickup / speed * 60))（与 6.2 相同的取整规则）。
2. 装货时间窗 load_time（若货源配置了 [开始, 结束] 墙钟）  
  - 若到达时刻 晚于 窗结束：接单失败，货源按规则下线。  
  - 若到达时刻 早于 窗开始：原地等待至窗开始，等待分钟数 = 窗开始仿真分钟 − 到达仿真分钟。
3. 运输耗时  
  - 使用货源字段 cost_time_minutes（整数分钟，含装卸与干线运输的设定口径）。  
  - 完成后司机位置更新为货源 end。
4. 仿真上界（若有）  
  - 若在「动身前」预估成功完单时刻会超过当月仿真上界，则该次接单不会产生收益（防止仿真结束前接无法完成的超长单）
休息（wait）：
- 直接按调用参数推进 duration_minutes，不改变位置。
## 1. 这版到底在解决什么

这道题不是普通的路径规划，也不是单纯让 LLM 每步选一个高价货。真正难点有三个：

1. 每个司机有自然语言偏好，偏好类型复杂，并且隐藏集文本未知。
2. 货源、时间、位置每一步都在变化，错误动作会影响整月收益。
3. 评分同时看净收益和偏好扣分，偏好错一次可能吃掉大量收益。

这版的核心判断是：

```text
偏好理解决定上限，确定性执行决定稳定性，经济打分决定收益。
```

所以它没有让 LLM 每一步直接决策，而是把系统拆成三层：

1. LLM 只负责把自然语言偏好编译成结构化规则。
2. 代码负责查询接口、构建候选事实、执行硬约束、维护台账。
3. 打分器把收益、时薪、配额、区域流动性、罚款风险合成一个 score，然后 argmax。

## 2. 文件结构

最高分提交主要关心这些文件：

```text
demo/agent/model_decision_service.py
demo/SUBMISSION.md
demo/tests/test_preference_execution.py
demo/check_compliance.sh
```

其中 `model_decision_service.py` 是全部决策逻辑，约 2600 行，基本是单文件架构。虽然文件长，但主线很清楚：

```text
decide(driver_id)
  -> get_driver_status / query_decision_history
  -> compile preferences once
  -> deterministic forced waits
  -> query cargo
  -> build candidate facts
  -> optional scout query
  -> optional guardian council
  -> deterministic score and choose action
```

`SUBMISSION.md` 是提交说明，解释设计原则和合规性。

`test_preference_execution.py` 锁定了几个关键行为：通用逐单罚款、硬禁规则、周末作息覆盖基础作息、每天回家门禁、月底配额紧迫度。

`check_compliance.sh` 用黑名单 grep 检查 agent 里有没有公开样例地名、品类、司机号等偏好常量，目的是证明代码不是靠硬编码公开集得分。

## 3. 总体架构

可以把这版理解成五层：

```text
L1 Preference Compiler
  自然语言 preferences -> machine_ir

L1.5 Policy Audit
  校验 LLM 编译结果，错的降级为 unknown，不带病执行

L2 Ledger
  维护本月已经接了什么、长途多少、每天几单、打卡天数等事实台账

L3 Market Observer
  调 query_cargo 获取候选，必要时扩查

L4 Deterministic Planner
  硬约束 veto + 经济 score + argmax

L5 Guardian Council
  只在高风险 unknown 偏好存在时触发，且只能调分/否决，不能越权指定动作
```

这套架构最重要的变化是：LLM 不再是每步最终拍板者，而是偏好编译器和少量未知风险审计器。大部分司机在偏好被结构化之后，后续都走确定性快车道。

## 4. 每轮决策主流程

入口函数是 `ModelDecisionService.decide(driver_id)`。

### 4.1 获取状态和历史

每轮先调用：

```text
get_driver_status(driver_id)
query_decision_history(driver_id, -1)
```

状态里拿当前位置、仿真时间、司机成本、当前可见偏好。历史用于维护 token 预算和最近动作。

代码还会缓存司机初始位置，这对“每天回家”类偏好很重要。如果偏好说要回家但没给坐标，就用首日初始点作为低置信 home fallback。

### 4.2 编译偏好

`_compiled_preference_policy` 会把当前可见 preferences 做 JSON 签名。如果签名没变，就直接用缓存；如果偏好变了，才调用 LLM 编译。

编译输出核心是 `machine_ir`，包括：

```text
rest_windows              休息/禁动时间窗
long_haul_limits          长途月度上限
cargo_targets             某月某货类最低配额
cargo_max_limits          某月某货类最多接几单
daily_order_caps          每天最多几单
off_day_requirements      每月至少几整天不出车
location_visit_targets    每月至少几天去某地
home_curfews              每天几点前回家，几点前不出车
order_rules               逐单罚款/硬禁规则
region_avoid/origin_avoid 地区规避
unknown_constraints       暂时无法结构化的偏好
```

这一步是全系统的地基。偏好只要被编译对，后续代码就能稳定执行；如果编译错，后面再强也会错。

### 4.3 查货前强制动作

真正查货前，代码会优先处理几个确定性红线：

```text
_forced_off_day_wait
_current_rest_wait_minutes
_home_curfew_decision
_pre_query_rest_guard
```

意思是：

1. 如果月末必须留整天不出车，就直接 wait 到当天结束。
2. 如果当前在休息/禁动窗内，就 wait 到窗尾。
3. 如果当前到了回家门禁阶段，就 wait 或 reposition 回家。
4. 如果现在查货会因为扫描耗时擦进休息窗，就不查货，提前 wait 睡穿。

这个设计非常关键。比赛里的 `query_cargo` 也会消耗仿真时间，所以“先查一下再说”可能导致在禁止活动窗口内产生扫描时间。这个版本专门用 pre-query guard 避免这种隐藏扣分。

### 4.4 市场查询和扩查

基础查货：

```text
query_cargo(k=220)
```

如果 token 已接近软上限，则降到 `k=120`。

查完货之后会重新取一次 status，因为查询本身推进了仿真时间。然后用 `_build_candidate_facts` 把货源转成候选事实。

如果候选太弱，或月底配额仍有缺口，会触发 `_run_deterministic_market_scout_queries` 做额外查询。扩查种子来自：

1. 欠额品类历史出现过的热点。
2. 当前高价值候选的起点/终点。
3. 司机历史观测过的高价货源热点。

这就是“区域价值/货源热点”的轻量实现：不直接读全量数据，只靠当前司机实际 query 到的货源逐步积累。

### 4.5 候选事实构建

`CandidateFact` 是候选货源的统一结构。每个候选都会算：

```text
cargo_id
pickup_km / pickup_min
haul_km
wait_min
transport_min
finish_min
price_yuan
cost_yuan
net_yuan_before_pref
net_per_hour_before_pref
near_end_cargo_seen
legal
veto_reasons
```

这一步把所有“事实”交给代码，不让 LLM 猜距离、时间、成本和是否合法。

合法性先看基础规则：

1. 是否晚于装货窗结束。
2. 是否货源已经 remove。
3. 是否完单超过仿真上界。
4. `cost_time_minutes` 是否有效。

偏好相关 veto 在后面统一处理。

## 5. 偏好编译系统

### 5.1 为什么要编译

自然语言偏好如果每步都让 LLM 重新读，会有三个问题：

1. token 爆炸。
2. 同一条偏好不同步骤解释不一致。
3. LLM 容易把“收益高”压过“偏好红线”。

所以这版让 LLM 只做一次编译，把文本变成可执行 policy。之后确定性代码读 policy。

### 5.2 编译提示词的重点

编译提示词专门针对比赛坑点写了大量约束：

1. 跨午夜窗口要向前归属，例如 23:00-06:00 的凌晨属于前一夜。
2. “周末晚 N 小时再休息”只顺延开始时间，不自动顺延结束时间。
3. “每天连续休息 N 小时”不能编成 start_hour == end_hour 的全天窗，而要落成同一日内 N 小时窗口。
4. 每日上限不能塞进月度槽位。
5. 整天歇车要编进 off_day_requirements。
6. 到某地至少 N 天要编进 location_visit_targets，而不是 cargo_targets。
7. 每天回家门禁要编进 home_curfews。
8. 逐单罚款要编成 order_rules 的 penalty，不要误编成 hard_veto。
9. 有补上月欠额和本月新指标时，要拆成两个目标，不要混成一个。

这部分其实是最高分版本的灵魂。它不是通用闲聊 prompt，而是针对官方评分规则和踩坑经验写出来的“偏好编译规范”。

### 5.3 编译后 audit

LLM 编译完不能直接信。`_audit_policy` 会做确定性审计，审不过就降级到 `unknown_constraints`。

主要审计包括：

1. `rest_windows` 必须有合法钟点，source_quote 能回到原文。
2. `cargo_targets` / `cargo_max_limits` 必须有合法 month、数量、货类。
3. 货类名要和运行时观测词表唯一对齐，防止“建材”对不上真实货源名。
4. 日/周周期不能误塞进月度槽位。
5. `long_haul_limits` 阈值和数量必须是正整数。
6. 地区硬禁如果和配额目标冲突，要剥离交给守护层。
7. `order_rules` 只能使用白名单字段和算子。
8. 有罚款的地区规则会优先保留为 penalty，不重复保留硬禁。
9. 已经被结构化字段覆盖的 unknown 会去重，避免每步无谓触发 Council。

审计原则是：

```text
错编比漏编危险。
不确定就降级 unknown，绝不硬执行。
```

## 6. 台账系统 Ledger

`_preference_ledger` 从已经选择的订单里构建事实记忆。

它记录：

```text
orders_total
cargo_name_counts_by_month
transport_duration_bins_by_month
transport_minutes_by_month
active_duration_bins_by_month
active_clock_hour_counts
orders_per_day_recent
order_count_by_day
last_order_start_min
```

这些台账用于：

1. 判断某月某货类已经接了几单。
2. 判断长途额度用了几单。
3. 判断每天最多 N 单是否已满。
4. 判断某天是否有活动，从而计算整天歇车。
5. 给 Guardian Council 提供事实背景。

特别重要的是 `transport_minutes_by_month`。它不是只存 over_8h 分箱，而是存原始 transport_min 列表。这样隐藏司机如果说“超过 6 小时最多 4 单”或“超过 10 小时最多 2 单”，代码可以按动态阈值重新计算。

## 7. 硬约束系统

硬约束统一在 `_deterministic_vetoes` 里判断。

一个候选会被 veto 的典型原因：

```text
rest_window_overlap
long_haul_quota_full
cargo_max_limit_full
daily_order_cap_full
region_avoid
order_rule_hard_veto
home_curfew_overlap
home_curfew_no_return
```

这里体现了一个很重要的设计边界：

1. 明确禁止、额度已满、休息窗冲突，直接 veto。
2. 有罚款但不绝对禁止的规则，不 veto，而是进入调价。
3. 配额最低目标不强制 veto 其他货，而是通过 bonus 提高目标货分数。

这避免了两个极端：

1. 偏好太硬，导致错过高收益单。
2. 偏好太软，导致偏好罚分爆炸。

## 8. 打分系统

候选最终由 `_candidate_score` 打分。

基础公式可以理解成：

```text
net = 原始净收益 - 逐单偏好罚款
nph = net / active_hours

score =
    net
  + 5.0 * nph
  + 12.0 * near_end_cargo_seen
  - 0.35 * pickup_km
  + quota_target_bonus
  + location_visit_bonus
  + council_adjustment
  - long_haul_soft_penalty
  - very_long_occupation_penalty
```

各项含义：

1. `net` 保证绝对收益。
2. `nph` 保证单位时间收益。
3. `near_end_cargo_seen` 是终点附近后续货源数量，近似区域价值。
4. `pickup_km` 惩罚赴装货空驶。
5. `quota_target_bonus` 负责最低配额履约。
6. `location_visit_bonus` 负责到某地打卡。
7. `council_adjustment` 是 LLM 守护层调分，但限幅。
8. 长途和超长占用有软惩罚。

只要最高分候选 `score > 0`，就接单。否则尝试受限 reposition；如果没有合适迁移，就 wait。

最高分提交里 `ltd_reposition` 默认关闭，所以大多数情况下无正分候选会 wait，而不是主动大空驶。这也是为了避免隐藏集里空驶/门禁/作息类偏好被主动迁移打爆。

## 9. 配额经济化

配额相关函数是 `_target_bonus`。

当某个候选货类命中 `cargo_targets`，且当前月该货类仍有 shortfall，就给它加 bonus。

核心变量：

```text
used       已接数量
min_count 目标数量
shortfall = min_count - used
penalty   少一单扣多少钱
lam       紧迫度，来自 shortfall / 估计剩余机会
```

bonus 近似：

```text
penalty * (1 + lam * shortfall) * days_left_factor * urgency_factor
```

其中：

1. `days_left_factor` 处理跨月补欠和月份紧迫感。
2. `lam` 是次梯度影子价格，表示“现在不接，后面还有多少机会补”。
3. `aggressive_fulfill` 开启时，只要进度落后线性配速，就把 `lam` 拉到 1，提前抢配额，不拖到月底。
4. 最高分提交新增 `urgency_factor`：当本月剩余天数已经接近 `shortfall + 3`，进一步提高目标货价值。

这就是提交名 `add deadline-aware quota urgency` 的含义。

这个补丁为什么有效：

1. 旧逻辑已经会补配额，但可能补得太晚。
2. 月底还差货时，错过一单目标货的代价很高。
3. 额外 urgency 只在临近 deadline 时触发，避免月初过度追低收益配额单。

## 10. 作息系统

作息相关函数主要是：

```text
_current_rest_wait_minutes
_rest_intervals_around
_merge_no_go_segments
_pre_query_rest_guard
_interval_overlaps_forbidden_window
```

它的几个关键细节：

1. 休息窗口会展开成仿真分钟区间。
2. 跨午夜窗口自动把结束时间加一天。
3. weekend / weekday 特殊窗口会覆盖重叠的 all-days 基础窗口。
4. 如果多个禁动窗口相邻或重叠，会合并成一个 no-go segment。
5. 如果当前在休息窗内，直接 wait 到段尾。
6. 如果查货扫描会擦进休息窗，提前 wait。
7. 候选订单的完整 active interval 与休息窗重叠，则直接 veto。

测试里专门验证了“周末窗口覆盖 all-days 基础窗口”：

```text
all:     21:00 -> 06:00
weekend: 23:00 -> 06:00
```

周末不能取并集成 21:00-06:00，否则会多休两小时，损失收益；也不能顺延到 23:00-08:00，否则会错过早晨合法运营时间。

## 11. 回家门禁系统

`home_curfews` 是这版后期很重要的泛化能力。

相关函数：

```text
_resolve_home_coord
_home_quiet_intervals_around
_home_curfew_decision
_cap_wait_for_home
```

它支持这种偏好：

```text
每天 X 点前回家，次日 Y 点前不接单/不空驶/不出车
```

执行逻辑：

1. 如果文本给 home 坐标，用坐标。
2. 如果没给坐标，用司机初始位置作为低置信家。
3. quiet 窗内一律 wait，不在窗内乱 reposition 反而可能制造更多罚分。
4. deadline 前如果按 ETA + 安全余量已经快赶不回家，就提前 reposition 回家。
5. 评估候选时，如果订单跨入 quiet 窗，或完成后回家来不及，直接 veto。

这类偏好很容易让普通收益模型翻车，因为它要求“接单前就预判完单后是否还来得及回家”。

## 12. 整天歇车和到某地打卡

### 12.1 整天歇车

函数：

```text
_active_day_set
_off_days_taken
_forced_off_day_wait
```

逻辑：

1. 一个订单的 active interval 只要覆盖某天，该天就不是整天歇车。
2. 月末如果剩余天数不足以完成 off-day 要求，就强制今天 wait 到结束。
3. 今天必须还是干净日，才能被用作整天歇车。

这对“每月至少 N 整天不出车”类偏好很关键。守护层无法可靠预留未来整天，所以必须确定性执行。

### 12.2 到某地打卡

函数：

```text
_location_visit_targets
_hits_visit_target
_visit_days_by_month
_visit_target_bonus
```

逻辑：

1. 文本有地名时，优先按货源起终点文本是否包含 keyword 判断。
2. 纯坐标任务才按终点进入半径判断。
3. 按天去重，不是按次数。
4. 月底缺打卡天数时，对真达标候选加更高 bonus。
5. 120km 内但没真正达标，只给弱引导，不计达标。

这个设计避免“以为到了附近就算打卡，官方仍然扣罚”的问题。

## 13. 通用逐单规则 order_rules

`order_rules` 把很多货源偏好统一成条件 + effect：

```text
field: cargo_name | start_region | end_region | either_region | deadhead_km | transport_minutes
op: contains | equals | gt | gte | lt | lte
effect: penalty | hard_veto
```

例如：

```text
赴装货空驶超过 55 公里扣 120
```

应该编成：

```json
{
  "field": "deadhead_km",
  "op": "gt",
  "value": 55,
  "effect": "penalty",
  "penalty_amount": 120
}
```

打分时不仅减 `120`，还会因为 net 下降导致 `5*nph` 一起下降，所以测试里断言分数下降是：

```text
120 + 5 * 120 / active_hours
```

这个很合理：罚款不仅减少总利润，也降低单位时间收益。

## 14. Guardian Council 的定位

`_preference_guardian_council` 不是主决策器。它只在 `unknown_constraints` 中存在 high risk 时触发。

触发后，Council 只看确定性 score 排名前 10 的候选。它可以输出：

```text
prefer
allow
soft_avoid
hard_avoid
must_wait
score_adjustment
```

但权限被严格限制：

1. hard_avoid 必须带依据，否则降级 soft_avoid。
2. score_adjustment 限幅在 -800 到 +800。
3. 未审候选在 Council 激活时不能参与 argmax。
4. Council 不能覆盖确定性作息休息。
5. 每司机每天最多调用 6 次，避免超时和 token 爆炸。

也就是说，LLM 只处理代码暂时表达不了的复杂偏好残差，不能凭空接管全部行动。

