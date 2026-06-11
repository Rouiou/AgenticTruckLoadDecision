#!/bin/sh
DIR="$(dirname "$0")/agent"
cd "$DIR" 2>/dev/null || { echo "找不到 agent 目录: $DIR"; exit 2; }
# 黑名单：初赛 D001/D002 + 复赛 D001 涉及的地名/品类/司机号/任务词
BLACKLIST='惠州|增城|深圳|四会|机械设备|蔬菜|盘库|寿宴|档口|赴宴|龙门吊|铸件|数码家电|水果|建材|何师傅|D00[0-9]'
HITS=$(grep -nE "$BLACKLIST" ./*.py 2>/dev/null)
if [ -n "$HITS" ]; then
  echo "❌ 合规门禁未通过：agent/ 内发现疑似偏好常量："
  echo "$HITS"; exit 1
fi
echo "✅ 合规门禁通过：agent/*.py 内无样例偏好常量(含复赛 水果/建材/何师傅)。"
exit 0
