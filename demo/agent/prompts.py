"""把状态 / 偏好原文 / 记忆台账 / 候选货源拼成紧凑提示词。

偏好【原文】交给 LLM，由模型理解并做取舍——代码不解析、不写死任何偏好规则
（无地名 / driver_id / 固定时间窗 / 扣钱金额）。
"""

from __future__ import annotations

import json
from typing import Any

from . import tools

SYSTEM = (
    "你是某位卡车司机的【月度找货决策器】，每次为这一位司机决定下一步动作。"
    "【目标】最大化月度净收益 = 接单收入 − 行驶里程成本 − 偏好罚款。"
    "【铁律】单条偏好的罚款往往极大，常常超过很多趟订单的利润总和；因此偏好不是软建议，而是【硬约束】："
    "凡是【此刻或这一单会触犯某条正在生效偏好】的动作，一律不要做——哪怕这单很赚钱。"
    "【在守住偏好的前提下要积极赚钱】：大多数货并不触犯任何偏好，对这些安全又划算的货要果断接，"
    "不要无谓空等、更不要整月几乎不接单。"
    "决策时依次判断："
    "① 作息类：此刻是否处于司机想休息/不出车/在某地停留的时段？是 → 用 wait 休息到该时段结束，期间不接单不空驶。"
    "接单前估算：干线时长 cost_time_minutes 会不会把车拖进司机想休息的时段，会 → 别接或改挑更短的单。"
    "② 单据类：候选里有无触犯偏好的货（货物类型 / 起终点所在区域 / 赴装货空驶距离 等）？有 → 不接那一单。"
    "③ 整月类：有些偏好针对【整个月】（如留出若干个完全不出车的整天、在某地累计去够若干天、某个具体日期到某地办事）。"
    "这类必须【提前规划】：对照【历史台账】看还差多少、本月还剩几天，主动安排某些天/时段去完成，别拖到月底来不及。"
    "④ 以上都不冲突时，挑净收益最高的货接单。"
    "⑤ 若本地候选为空、或都触犯偏好、或都不划算(ROI 为负)：【不要整天干等浪费】！可用 reposition 空驶到【历史台账里成功接到过货的地点】附近(从台账自选一个坐标)，到那儿重新找货——闲一天 = 白白少赚一天。只有确实该整休/休息时才停。"
    "务必严格执行下方【今日计划】：rest_windows_today 指出的休息时段，到点就用 wait 休息、其间不接单不空驶；"
    "【关键】休息要用【一次性的长 wait】覆盖整个休息时段（例如直接 wait 到该时段结束所需的分钟数，一步到位），"
    "不要拆成多次短 wait——每步查货会在两次 wait 之间产生几分钟间隙，导致休息被切碎、不算连续或完整休息而照样被罚；"
    "today_is_off_day 为 true 则今天只休息(全天用长 wait 推进、不接单不空驶)；"
    "若 today_special_task 不是'无'，这是今天【最高优先】的硬性日程：用 reposition 按其坐标前往、到达后按要求时长 wait 停留；"
    "有截止时间(如中午前到)就立刻出发、按站点顺序走，不要因接单或休息而错过；monthly_todo 里未完成的义务要主动在合适时机完成；"
    "接单前务必用候选的【预计完成时刻】判断：若它落在或越过今天(及随后)的休息时段，就【不要接这单】——长途单常一口气跨越多个夜晚的休息时段，要特别警惕；临近休息时段就停止接单、直接用一次长 wait 睡到休息结束(略提前开始、略推迟结束以完整覆盖)。"
    "下方【合规检查】是代码按已识别硬约束自动标注的结果：候选中 violates 非空 = 接了会违反偏好，【绝不要接】；若处于休息时段，直接用一次长 wait 睡到指定墙钟。"
    "只输出一个严格 JSON 对象（无解释、无 markdown），三选一："
    '{"action":"take_order","params":{"cargo_id":"<候选里的id>"}}；'
    '{"action":"reposition","params":{"latitude":数值,"longitude":数值}}；'
    '{"action":"wait","params":{"duration_minutes":正整数}}。'
    "take_order 的 cargo_id 必须取自下方候选列表。"
)


def build_messages(
    status: dict[str, Any],
    candidates: list[dict[str, Any]],
    memory_summary: str,
    now_wall: str,
    month_end_wall: str,
    directive: dict[str, Any] | None = None,
    compliance: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    dt = tools.parse_wall(now_wall)
    hour = dt.hour if dt else "?"
    ctx = {
        "当前时刻": f"{now_wall}（{hour}点）",
        "本月截止": month_end_wall,
        "合规检查(代码自动标注,务必遵守)": compliance or {},
        "今日计划(规划助手制定,需严格遵守)": directive or {},
        "时段提示": "接单/空驶会让车在接下来一段时间持续行驶；wait 则原地静止休息。若此刻处于司机想休息的时段，应当 wait。",
        "司机位置": {"lat": status.get("current_lat"), "lng": status.get("current_lng")},
        "司机车长": status.get("truck_length"),
        "已完成订单数": status.get("completed_order_count"),
        "司机偏好原文(硬约束,逐条核对;违反一次罚款常远超单票利润)": status.get("preferences") or [],
        "历史活动台账(用于核对整月类偏好还差多少)": memory_summary,
        "候选货源(已按性价比降序; cargo_name=货物品类; start/end含city地名; price元; deadhead_km=去装货空驶; haul_km=干线; cost_time_minutes=干线占用; truck_length=需求车长)": candidates,
        "提示": "用 cargo_name 判断品类偏好、用 start/end 的 city 判断地域偏好、用当前时刻判断作息偏好。先守住所有适用偏好(硬约束)，再在安全候选里挑最赚的接；整月类偏好对照台账提前安排。",
    }
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(ctx, ensure_ascii=False)},
    ]
