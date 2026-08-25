#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"$ROOT/scripts/run_python.sh" "$ROOT/scripts/train_gridworld.py"
"$ROOT/scripts/run_python.sh" "$ROOT/scripts/train_cartpole.py" --steps 150000
"$ROOT/scripts/run_python.sh" "$ROOT/scripts/train_policy_gradient.py" --episodes 600
"$ROOT/scripts/run_python.sh" "$ROOT/scripts/train_actor_critic.py" --steps 300000
"$ROOT/scripts/run_python.sh" "$ROOT/scripts/train_ppo.py" --steps 120000
"$ROOT/scripts/run_python.sh" "$ROOT/scripts/train_continuous_ppo.py" --steps 200000
"$ROOT/scripts/run_python.sh" "$ROOT/scripts/train_sac.py" --steps 100000
"$ROOT/scripts/run_python.sh" "$ROOT/scripts/generate_foundations.py"
"$ROOT/scripts/run_python.sh" "$ROOT/scripts/generate_advanced_lessons.py"
"$ROOT/scripts/run_python.sh" "$ROOT/scripts/generate_td_variants.py"
"$ROOT/scripts/run_python.sh" "$ROOT/scripts/generate_exploration.py"
"$ROOT/scripts/run_python.sh" "$ROOT/scripts/generate_imitation.py"
"$ROOT/scripts/run_python.sh" "$ROOT/scripts/generate_debug_lessons.py"

echo "全部训练与课程实验数据生成完成。运行 ./scripts/serve.sh 后打开 http://127.0.0.1:8000"
