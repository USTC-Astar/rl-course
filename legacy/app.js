"use strict";

const state = {
  gridData: null,
  gridRouteIndex: 0,
  gridTimer: null,
  gridSnapshotIndex: 0,
  selectedGridState: null,
  cartpoleData: null,
  cartpoleFrame: 0,
  cartpoleTimer: null,
  cartpolePlaying: true,
  policyData: null,
  policyFrame: 0,
  policyTimer: null,
  policyPlaying: true,
  actorCriticData: null,
  actorCriticFrame: 0,
  actorCriticTimer: null,
  actorCriticPlaying: true,
  ppoData: null,
  ppoFrame: 0,
  ppoTimer: null,
  ppoPlaying: true,
  continuousData: null,
  continuousFrame: 0,
  continuousTimer: null,
  continuousPlaying: true,
  sacData: null,
  sacFrame: 0,
  sacTimer: null,
  sacPlaying: true,
};

document.addEventListener("DOMContentLoaded", async () => {
  setupNavigation();
  setupControls();
  await Promise.all([
    loadGridWorld(),
    loadCartPole(),
    loadPolicyGradient(),
    loadActorCritic(),
    loadPPO(),
    loadContinuousPPO(),
    loadSAC(),
  ]);
  window.addEventListener("resize", redrawCharts);
});

function setupNavigation() {
  const links = [...document.querySelectorAll(".sidebar nav a")];
  const sections = links
    .map((link) => document.querySelector(link.getAttribute("href")))
    .filter(Boolean);

  const observer = new IntersectionObserver(
    (entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (!visible) return;
      links.forEach((link) => {
        link.classList.toggle("active", link.getAttribute("href") === `#${visible.target.id}`);
      });
    },
    { rootMargin: "-20% 0px -65% 0px", threshold: [0.05, 0.2, 0.5] }
  );
  sections.forEach((section) => observer.observe(section));
}

function setupControls() {
  document.querySelector("#grid-reset").addEventListener("click", () => {
    stopGridPlayback();
    state.gridRouteIndex = 0;
    renderGrid();
  });
  document.querySelector("#grid-step").addEventListener("click", stepGrid);
  document.querySelector("#grid-play").addEventListener("click", toggleGridPlayback);
  document.querySelector("#snapshot-slider").addEventListener("input", (event) => {
    state.gridSnapshotIndex = Number(event.target.value);
    renderGrid();
  });
  document.querySelector("#cartpole-reset").addEventListener("click", () => {
    state.cartpoleFrame = 0;
    state.cartpolePlaying = true;
    document.querySelector("#cartpole-play").textContent = "暂停";
    startCartPolePlayback();
  });
  document.querySelector("#cartpole-play").addEventListener("click", () => {
    state.cartpolePlaying = !state.cartpolePlaying;
    document.querySelector("#cartpole-play").textContent = state.cartpolePlaying ? "暂停" : "继续";
    if (state.cartpolePlaying) startCartPolePlayback();
    else stopCartPolePlayback();
  });
  document.querySelector("#policy-reset").addEventListener("click", () => {
    state.policyFrame = 0;
    state.policyPlaying = true;
    document.querySelector("#policy-play").textContent = "暂停";
    startPolicyPlayback();
    drawPolicyFrame();
  });
  document.querySelector("#policy-play").addEventListener("click", () => {
    state.policyPlaying = !state.policyPlaying;
    document.querySelector("#policy-play").textContent = state.policyPlaying ? "暂停" : "继续";
    if (state.policyPlaying) startPolicyPlayback();
    else stopPolicyPlayback();
  });
  document.querySelector("#actor-critic-reset").addEventListener("click", () => {
    state.actorCriticFrame = 0;
    state.actorCriticPlaying = true;
    document.querySelector("#actor-critic-play").textContent = "暂停";
    startActorCriticPlayback();
    drawActorCriticFrame();
  });
  document.querySelector("#actor-critic-play").addEventListener("click", () => {
    state.actorCriticPlaying = !state.actorCriticPlaying;
    document.querySelector("#actor-critic-play").textContent = state.actorCriticPlaying ? "暂停" : "继续";
    if (state.actorCriticPlaying) startActorCriticPlayback();
    else stopActorCriticPlayback();
  });
  document.querySelector("#ppo-reset").addEventListener("click", () => {
    state.ppoFrame = 0;
    state.ppoPlaying = true;
    document.querySelector("#ppo-play").textContent = "暂停";
    startPPOPlayback();
    drawPPOFrame();
  });
  document.querySelector("#ppo-play").addEventListener("click", () => {
    state.ppoPlaying = !state.ppoPlaying;
    document.querySelector("#ppo-play").textContent = state.ppoPlaying ? "暂停" : "继续";
    if (state.ppoPlaying) startPPOPlayback();
    else stopPPOPlayback();
  });
  document.querySelector("#continuous-reset").addEventListener("click", () => {
    state.continuousFrame = 0;
    state.continuousPlaying = true;
    document.querySelector("#continuous-play").textContent = "暂停";
    startContinuousPlayback();
    drawContinuousFrame();
  });
  document.querySelector("#continuous-play").addEventListener("click", () => {
    state.continuousPlaying = !state.continuousPlaying;
    document.querySelector("#continuous-play").textContent = state.continuousPlaying ? "暂停" : "继续";
    if (state.continuousPlaying) startContinuousPlayback();
    else stopContinuousPlayback();
  });
  document.querySelector("#sac-reset").addEventListener("click", () => {
    state.sacFrame = 0;
    state.sacPlaying = true;
    document.querySelector("#sac-play").textContent = "暂停";
    startSACPlayback();
    drawSACFrame();
  });
  document.querySelector("#sac-play").addEventListener("click", () => {
    state.sacPlaying = !state.sacPlaying;
    document.querySelector("#sac-play").textContent = state.sacPlaying ? "暂停" : "继续";
    if (state.sacPlaying) startSACPlayback();
    else stopSACPlayback();
  });
}

async function loadJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

async function loadGridWorld() {
  const status = document.querySelector("#grid-status");
  try {
    state.gridData = await loadJson("data/gridworld.json");
    state.selectedGridState = positionToState(state.gridData.environment.start);
    const slider = document.querySelector("#snapshot-slider");
    slider.max = Math.max(0, state.gridData.training.snapshots.length - 1);
    slider.value = slider.max;
    state.gridSnapshotIndex = Number(slider.value);
    status.className = "data-status ready";
    status.textContent = "训练数据读取成功。下面的路线和 Q 值来自本机实际训练。";
    renderGrid();
    renderGridMetrics();
    drawGridRewardChart();
  } catch (error) {
    status.className = "data-status error";
    status.textContent = `没有找到网格训练数据。请运行 ./scripts/run_python.sh scripts/train_gridworld.py。错误：${error.message}`;
  }
}

function renderGrid() {
  if (!state.gridData) return;
  const grid = document.querySelector("#grid-world");
  const env = state.gridData.environment;
  const learned = state.gridData.learned;
  const snapshot = state.gridData.training.snapshots[state.gridSnapshotIndex];
  const robotPosition = learned.route[Math.min(state.gridRouteIndex, learned.route.length - 1)];
  grid.style.gridTemplateColumns = `repeat(${env.cols}, 1fr)`;
  grid.replaceChildren();

  for (let row = 0; row < env.rows; row += 1) {
    for (let col = 0; col < env.cols; col += 1) {
      const position = [row, col];
      const cell = document.createElement("button");
      const stateIndex = positionToState(position);
      cell.className = "grid-cell";
      cell.type = "button";
      cell.dataset.state = stateIndex;

      if (samePosition(position, env.walls)) cell.classList.add("wall");
      else if (samePosition(position, env.traps)) cell.classList.add("trap");
      else if (equalPosition(position, env.goal)) cell.classList.add("goal");
      if (equalPosition(position, env.start)) cell.classList.add("start");
      if (state.selectedGridState === stateIndex) cell.classList.add("selected");

      let symbol = snapshot?.policy?.[row]?.[col] ?? "";
      if (symbol === "goal") symbol = "★";
      else if (symbol === "trap") symbol = "⚠";
      else if (symbol === "wall") symbol = "";
      const policy = document.createElement("span");
      policy.className = "policy-arrow";
      policy.textContent = symbol;
      cell.append(policy);

      const routeVisit = learned.route.findIndex((item) => equalPosition(item, position));
      if (routeVisit >= 0) {
        const marker = document.createElement("small");
        marker.className = "route-index";
        marker.textContent = routeVisit;
        cell.append(marker);
      }

      if (equalPosition(position, robotPosition)) {
        const robot = document.createElement("span");
        robot.className = "robot";
        robot.textContent = "🤖";
        cell.append(robot);
      }

      const blocked = cell.classList.contains("wall") || cell.classList.contains("trap") || cell.classList.contains("goal");
      cell.disabled = cell.classList.contains("wall");
      if (!blocked) {
        cell.addEventListener("click", () => {
          state.selectedGridState = stateIndex;
          renderGrid();
        });
      }
      grid.append(cell);
    }
  }

  const label = document.querySelector("#snapshot-label");
  label.textContent = snapshot ? `第 ${snapshot.episode} 回合 · ε=${snapshot.epsilon}` : "无快照";
  renderQBars();
}

function renderQBars() {
  if (!state.gridData || state.selectedGridState === null) return;
  const env = state.gridData.environment;
  const position = stateToPosition(state.selectedGridState, env.cols);
  const values = state.gridData.learned.q_table[state.selectedGridState];
  const maxMagnitude = Math.max(...values.map((value) => Math.abs(value)), 0.001);
  document.querySelector("#selected-cell").textContent = `位置：第 ${position[0] + 1} 行，第 ${position[1] + 1} 列`;
  document.querySelectorAll("#q-bars > div").forEach((bar, index) => {
    const value = values[index];
    bar.querySelector("i").style.setProperty("--bar-width", `${Math.abs(value) / maxMagnitude * 100}%`);
    bar.querySelector("i").style.filter = value < 0 ? "hue-rotate(125deg) saturate(1.8)" : "none";
    bar.querySelector("b").textContent = value.toFixed(3);
  });
}

function stepGrid() {
  if (!state.gridData) return;
  const lastIndex = state.gridData.learned.route.length - 1;
  state.gridRouteIndex = state.gridRouteIndex >= lastIndex ? 0 : state.gridRouteIndex + 1;
  renderGrid();
}

function toggleGridPlayback() {
  if (state.gridTimer) stopGridPlayback();
  else {
    document.querySelector("#grid-play").textContent = "暂停播放";
    state.gridTimer = window.setInterval(stepGrid, 500);
  }
}

function stopGridPlayback() {
  if (state.gridTimer) window.clearInterval(state.gridTimer);
  state.gridTimer = null;
  document.querySelector("#grid-play").textContent = "自动播放";
}

function renderGridMetrics() {
  const learned = state.gridData.learned;
  document.querySelector("#grid-metrics").innerHTML = [
    `<span>路线 ${learned.route.length - 1} 步</span>`,
    `<span>成功率 ${(learned.success_rate * 100).toFixed(0)}%</span>`,
    `<span>末 100 回合均值 ${learned.average_recent_reward.toFixed(2)}</span>`,
  ].join("");
}

async function loadCartPole() {
  const status = document.querySelector("#cartpole-status");
  try {
    state.cartpoleData = await loadJson("data/cartpole.json");
    status.className = "data-status ready";
    status.textContent = "训练数据读取成功。动画使用评估阶段表现最好的一次真实状态轨迹。";
    renderCartPoleMetrics();
    drawCartPoleRewardChart();
    startCartPolePlayback();
  } catch (error) {
    status.className = "data-status error";
    status.textContent = `没有找到 CartPole 训练数据。请运行 ./scripts/run_python.sh scripts/train_cartpole.py --steps 150000。错误：${error.message}`;
    drawCartPolePlaceholder();
  }
}

function startCartPolePlayback() {
  if (!state.cartpoleData || state.cartpoleTimer) return;
  drawCartPoleFrame();
  state.cartpoleTimer = window.setInterval(() => {
    if (!state.cartpolePlaying) return;
    const trajectory = state.cartpoleData.evaluation.trajectory;
    state.cartpoleFrame = (state.cartpoleFrame + 1) % trajectory.length;
    drawCartPoleFrame();
  }, 35);
}

function stopCartPolePlayback() {
  if (state.cartpoleTimer) window.clearInterval(state.cartpoleTimer);
  state.cartpoleTimer = null;
}

function drawCartPolePlaceholder() {
  drawCanvasMessage(
    document.querySelector("#cartpole-canvas"),
    "完成 CartPole 训练后，这里会显示小车动画。"
  );
}

function drawCanvasMessage(canvas, message) {
  const context = canvas.getContext("2d");
  fitCanvas(canvas);
  context.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  context.fillStyle = "#64706a";
  context.textAlign = "center";
  context.font = "16px sans-serif";
  context.fillText(message, canvas.clientWidth / 2, canvas.clientHeight / 2);
}

function drawCartPoleFrame() {
  if (!state.cartpoleData) return;
  const trajectory = state.cartpoleData.evaluation.trajectory;
  const frame = trajectory[state.cartpoleFrame];
  const [x, xDot, theta, thetaDot] = frame;
  const canvas = document.querySelector("#cartpole-canvas");
  paintCartPoleState(canvas, frame, "#136f55");

  document.querySelector("#state-x").textContent = x.toFixed(3);
  document.querySelector("#state-x-dot").textContent = xDot.toFixed(3);
  document.querySelector("#state-theta").textContent = `${theta.toFixed(3)} rad`;
  document.querySelector("#state-theta-dot").textContent = thetaDot.toFixed(3);
  document.querySelector("#cartpole-progress").style.width = `${(state.cartpoleFrame + 1) / trajectory.length * 100}%`;
}

function paintCartPoleState(canvas, frame, cartColor) {
  const [x, , theta] = frame;
  const context = canvas.getContext("2d");
  fitCanvas(canvas);
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  context.clearRect(0, 0, width, height);

  const groundY = height * 0.78;
  context.strokeStyle = "#7c8d86";
  context.lineWidth = 3;
  context.beginPath();
  context.moveTo(35, groundY);
  context.lineTo(width - 35, groundY);
  context.stroke();

  const centerX = width / 2 + (x / 2.4) * width * 0.38;
  const cartWidth = Math.max(70, width * 0.13);
  const cartHeight = 35;
  context.fillStyle = cartColor;
  roundedRect(context, centerX - cartWidth / 2, groundY - cartHeight, cartWidth, cartHeight, 8);
  context.fill();

  context.fillStyle = "#17231f";
  context.beginPath();
  context.arc(centerX - cartWidth * 0.3, groundY + 6, 10, 0, Math.PI * 2);
  context.arc(centerX + cartWidth * 0.3, groundY + 6, 10, 0, Math.PI * 2);
  context.fill();

  const pivotY = groundY - cartHeight;
  const poleLength = Math.min(155, height * 0.46);
  const poleEndX = centerX + Math.sin(theta) * poleLength;
  const poleEndY = pivotY - Math.cos(theta) * poleLength;
  context.strokeStyle = Math.abs(theta) > 0.15 ? "#d85c52" : "#f2bd4b";
  context.lineWidth = 10;
  context.lineCap = "round";
  context.beginPath();
  context.moveTo(centerX, pivotY);
  context.lineTo(poleEndX, poleEndY);
  context.stroke();
  context.fillStyle = "#17231f";
  context.beginPath();
  context.arc(centerX, pivotY, 8, 0, Math.PI * 2);
  context.fill();
}

function renderCartPoleMetrics() {
  const data = state.cartpoleData;
  document.querySelector("#cartpole-metrics").innerHTML = [
    `<span>训练 ${data.config.total_steps} 步</span>`,
    `<span>平均评估 ${data.evaluation.average_return.toFixed(1)}</span>`,
    `<span>最佳 ${data.evaluation.best_return.toFixed(0)}</span>`,
    `<span>参数 ${data.parameter_count.toLocaleString()}</span>`,
  ].join("");
}

async function loadPolicyGradient() {
  const status = document.querySelector("#policy-status");
  try {
    state.policyData = await loadJson("data/policy_gradient.json");
    status.className = "data-status ready";
    status.textContent = "训练数据读取成功。动画会同步显示策略网络当时输出的左右动作概率。";
    renderPolicyMetrics();
    drawPolicyRewardChart();
    startPolicyPlayback();
  } catch (error) {
    status.className = "data-status error";
    status.textContent = `没有找到策略梯度数据。请运行 ./scripts/run_python.sh scripts/train_policy_gradient.py --episodes 600。错误：${error.message}`;
    drawCanvasMessage(
      document.querySelector("#policy-canvas"),
      "完成策略梯度训练后，这里会显示动作概率动画。"
    );
  }
}

function startPolicyPlayback() {
  if (!state.policyData || state.policyTimer) return;
  drawPolicyFrame();
  state.policyTimer = window.setInterval(() => {
    if (!state.policyPlaying) return;
    const trajectory = state.policyData.evaluation.trajectory;
    state.policyFrame = (state.policyFrame + 1) % trajectory.length;
    drawPolicyFrame();
  }, 35);
}

function stopPolicyPlayback() {
  if (state.policyTimer) window.clearInterval(state.policyTimer);
  state.policyTimer = null;
}

function drawPolicyFrame() {
  if (!state.policyData) return;
  const evaluation = state.policyData.evaluation;
  const frameIndex = Math.min(state.policyFrame, evaluation.trajectory.length - 1);
  const frame = evaluation.trajectory[frameIndex];
  const probabilities = evaluation.action_probabilities[frameIndex];
  const action = evaluation.actions[frameIndex];
  paintCartPoleState(document.querySelector("#policy-canvas"), frame, "#16324f");

  const leftPercent = probabilities[0] * 100;
  const rightPercent = probabilities[1] * 100;
  document.querySelector("#policy-left-value").textContent = `${leftPercent.toFixed(1)}%`;
  document.querySelector("#policy-right-value").textContent = `${rightPercent.toFixed(1)}%`;
  document.querySelector("#policy-left-bar").style.width = `${leftPercent}%`;
  document.querySelector("#policy-right-bar").style.width = `${rightPercent}%`;
  document.querySelector("#policy-action").textContent = action === 0 ? "← 向左推动" : "向右推动 →";
  document.querySelector("#policy-progress").style.width = `${(frameIndex + 1) / evaluation.trajectory.length * 100}%`;
}

function renderPolicyMetrics() {
  const data = state.policyData;
  document.querySelector("#policy-metrics").innerHTML = [
    `<span>最大 ${data.config.episodes} 回合</span>`,
    `<span>实际 ${data.episodes_completed} 回合</span>`,
    `<span>平均评估 ${data.evaluation.average_return.toFixed(1)}</span>`,
    `<span>并行环境 ${data.config.vector_env_count}</span>`,
  ].join("");
}

async function loadActorCritic() {
  const status = document.querySelector("#actor-critic-status");
  try {
    state.actorCriticData = await loadJson("data/actor_critic.json");
    status.className = "data-status ready";
    status.textContent = "训练数据读取成功。每一帧都会同时展示行动者概率与评论家价值判断。";
    renderActorCriticMetrics();
    drawActorCriticRewardChart();
    startActorCriticPlayback();
  } catch (error) {
    status.className = "data-status error";
    status.textContent = `没有找到行动者-评论家数据。请运行 ./scripts/run_python.sh scripts/train_actor_critic.py --steps 300000。错误：${error.message}`;
    drawCanvasMessage(
      document.querySelector("#actor-critic-canvas"),
      "完成行动者-评论家训练后，这里会显示双角色动画。"
    );
  }
}

function startActorCriticPlayback() {
  if (!state.actorCriticData || state.actorCriticTimer) return;
  drawActorCriticFrame();
  state.actorCriticTimer = window.setInterval(() => {
    if (!state.actorCriticPlaying) return;
    const trajectory = state.actorCriticData.evaluation.trajectory;
    state.actorCriticFrame = (state.actorCriticFrame + 1) % trajectory.length;
    drawActorCriticFrame();
  }, 35);
}

function stopActorCriticPlayback() {
  if (state.actorCriticTimer) window.clearInterval(state.actorCriticTimer);
  state.actorCriticTimer = null;
}

function drawActorCriticFrame() {
  if (!state.actorCriticData) return;
  const evaluation = state.actorCriticData.evaluation;
  const frameIndex = Math.min(
    state.actorCriticFrame,
    evaluation.trajectory.length - 1
  );
  const frame = evaluation.trajectory[frameIndex];
  const probabilities = evaluation.action_probabilities[frameIndex];
  const predictedValue = evaluation.state_values[frameIndex];
  const realizedReturn = evaluation.realized_returns[frameIndex];
  const advantage = realizedReturn - predictedValue;
  const action = evaluation.actions[frameIndex];
  paintCartPoleState(
    document.querySelector("#actor-critic-canvas"),
    frame,
    "#6f4aa8"
  );

  const leftPercent = probabilities[0] * 100;
  const rightPercent = probabilities[1] * 100;
  document.querySelector("#ac-left-value").textContent = `${leftPercent.toFixed(1)}%`;
  document.querySelector("#ac-right-value").textContent = `${rightPercent.toFixed(1)}%`;
  document.querySelector("#ac-left-bar").style.width = `${leftPercent}%`;
  document.querySelector("#ac-right-bar").style.width = `${rightPercent}%`;
  document.querySelector("#ac-value").textContent = predictedValue.toFixed(2);
  document.querySelector("#ac-return").textContent = realizedReturn.toFixed(2);
  const advantageElement = document.querySelector("#ac-advantage");
  advantageElement.textContent = `${advantage >= 0 ? "+" : ""}${advantage.toFixed(2)}`;
  advantageElement.parentElement.classList.toggle("positive", advantage >= 0);
  advantageElement.parentElement.classList.toggle("negative", advantage < 0);
  document.querySelector("#ac-action").textContent = action === 0 ? "← 向左推动" : "向右推动 →";
  document.querySelector("#ac-progress").style.width = `${(frameIndex + 1) / evaluation.trajectory.length * 100}%`;
}

function renderActorCriticMetrics() {
  const data = state.actorCriticData;
  document.querySelector("#actor-critic-metrics").innerHTML = [
    `<span>实际 ${data.steps_completed} 步</span>`,
    `<span>更新 ${data.update_count} 次</span>`,
    `<span>平均评估 ${data.evaluation.average_return.toFixed(1)}</span>`,
    `<span>rollout ${data.config.rollout_steps} 步</span>`,
  ].join("");
}

async function loadPPO() {
  const status = document.querySelector("#ppo-status");
  try {
    state.ppoData = await loadJson("data/ppo.json");
    status.className = "data-status ready";
    status.textContent = "训练数据读取成功。实验台同时展示最终策略、评论家判断和训练限速诊断。";
    renderPPOMetrics();
    renderPPODiagnostics();
    drawPPORewardChart();
    startPPOPlayback();
  } catch (error) {
    status.className = "data-status error";
    status.textContent = `没有找到 PPO 数据。请运行 ./scripts/run_python.sh scripts/train_ppo.py --steps 120000。错误：${error.message}`;
    drawCanvasMessage(
      document.querySelector("#ppo-canvas"),
      "完成 PPO 训练后，这里会显示策略限速实验。"
    );
  }
}

function startPPOPlayback() {
  if (!state.ppoData || state.ppoTimer) return;
  drawPPOFrame();
  state.ppoTimer = window.setInterval(() => {
    if (!state.ppoPlaying) return;
    const trajectory = state.ppoData.evaluation.trajectory;
    state.ppoFrame = (state.ppoFrame + 1) % trajectory.length;
    drawPPOFrame();
  }, 35);
}

function stopPPOPlayback() {
  if (state.ppoTimer) window.clearInterval(state.ppoTimer);
  state.ppoTimer = null;
}

function drawPPOFrame() {
  if (!state.ppoData) return;
  const evaluation = state.ppoData.evaluation;
  const frameIndex = Math.min(state.ppoFrame, evaluation.trajectory.length - 1);
  const frame = evaluation.trajectory[frameIndex];
  const probabilities = evaluation.action_probabilities[frameIndex];
  const predictedValue = evaluation.state_values[frameIndex];
  const realizedReturn = evaluation.realized_returns[frameIndex];
  const advantage = realizedReturn - predictedValue;
  const action = evaluation.actions[frameIndex];
  paintCartPoleState(document.querySelector("#ppo-canvas"), frame, "#b16b24");

  const leftPercent = probabilities[0] * 100;
  const rightPercent = probabilities[1] * 100;
  document.querySelector("#ppo-left-value").textContent = `${leftPercent.toFixed(1)}%`;
  document.querySelector("#ppo-right-value").textContent = `${rightPercent.toFixed(1)}%`;
  document.querySelector("#ppo-left-bar").style.width = `${leftPercent}%`;
  document.querySelector("#ppo-right-bar").style.width = `${rightPercent}%`;
  document.querySelector("#ppo-value").textContent = predictedValue.toFixed(2);
  document.querySelector("#ppo-return").textContent = realizedReturn.toFixed(2);
  const advantageElement = document.querySelector("#ppo-advantage");
  advantageElement.textContent = `${advantage >= 0 ? "+" : ""}${advantage.toFixed(2)}`;
  advantageElement.parentElement.classList.toggle("positive", advantage >= 0);
  advantageElement.parentElement.classList.toggle("negative", advantage < 0);
  document.querySelector("#ppo-action").textContent = action === 0 ? "← 向左推动" : "向右推动 →";
  document.querySelector("#ppo-progress").style.width = `${(frameIndex + 1) / evaluation.trajectory.length * 100}%`;
}

function renderPPOMetrics() {
  const data = state.ppoData;
  document.querySelector("#ppo-metrics").innerHTML = [
    `<span>实际 ${data.steps_completed} 步</span>`,
    `<span>更新 ${data.update_count} 次</span>`,
    `<span>平均评估 ${data.evaluation.average_return.toFixed(1)}</span>`,
    `<span>每批 ${data.config.update_epochs} 轮</span>`,
  ].join("");
}

function renderPPODiagnostics() {
  const data = state.ppoData;
  const averageClipFraction = data.clip_fraction_history.reduce(
    (sum, value) => sum + value,
    0
  ) / Math.max(1, data.clip_fraction_history.length);
  const finalKL = data.approximate_kl_history.at(-1) ?? 0;
  document.querySelector("#ppo-clip-coefficient").textContent = data.config.clip_coefficient.toFixed(2);
  document.querySelector("#ppo-clip-fraction").textContent = `${(averageClipFraction * 100).toFixed(1)}%`;
  document.querySelector("#ppo-kl").textContent = finalKL.toFixed(4);
  document.querySelector("#ppo-epochs").textContent = data.config.update_epochs;
}

async function loadContinuousPPO() {
  const status = document.querySelector("#continuous-status");
  try {
    state.continuousData = await loadJson("data/continuous_ppo.json");
    status.className = "data-status ready";
    status.textContent = "训练数据读取成功。动画使用最佳模型的确定性中心动作，并同步显示高斯策略宽度。";
    renderContinuousMetrics();
    renderContinuousDiagnostics();
    drawContinuousRewardChart();
    startContinuousPlayback();
  } catch (error) {
    status.className = "data-status error";
    status.textContent = `没有找到连续动作 PPO 数据。请运行 ./scripts/run_python.sh scripts/train_continuous_ppo.py --steps 200000。错误：${error.message}`;
    drawCanvasMessage(
      document.querySelector("#continuous-canvas"),
      "完成 MuJoCo 连续动作训练后，这里会显示倒立摆动画。"
    );
    drawCanvasMessage(
      document.querySelector("#continuous-gaussian-canvas"),
      "完成训练后，这里会显示动作分布。"
    );
  }
}

function startContinuousPlayback() {
  if (!state.continuousData || state.continuousTimer) return;
  drawContinuousFrame();
  state.continuousTimer = window.setInterval(() => {
    if (!state.continuousPlaying) return;
    const trajectory = state.continuousData.evaluation.trajectory;
    state.continuousFrame = (state.continuousFrame + 1) % trajectory.length;
    drawContinuousFrame();
  }, 30);
}

function stopContinuousPlayback() {
  if (state.continuousTimer) window.clearInterval(state.continuousTimer);
  state.continuousTimer = null;
}

function drawContinuousFrame() {
  if (!state.continuousData) return;
  const data = state.continuousData;
  const evaluation = data.evaluation;
  const frameIndex = Math.min(
    state.continuousFrame,
    evaluation.trajectory.length - 1
  );
  const [position, angle, velocity, angularVelocity] = evaluation.trajectory[frameIndex];
  const action = evaluation.actions[frameIndex][0];
  const actionMean = evaluation.action_means[frameIndex][0];
  const actionStandardDeviation = evaluation.action_standard_deviations[frameIndex][0];
  const predictedValue = evaluation.state_values[frameIndex];
  const realizedReturn = evaluation.realized_returns[frameIndex];
  const advantage = realizedReturn - predictedValue;
  const actionLow = data.action_low[0];
  const actionHigh = data.action_high[0];

  paintContinuousPendulum(
    document.querySelector("#continuous-canvas"),
    [position, angle, velocity, angularVelocity],
    action,
    actionLow,
    actionHigh
  );
  drawContinuousGaussian(
    document.querySelector("#continuous-gaussian-canvas"),
    actionMean,
    actionStandardDeviation,
    action,
    actionLow,
    actionHigh
  );

  document.querySelector("#continuous-force").textContent = formatSigned(action, 3);
  document.querySelector("#continuous-mean").textContent = formatSigned(actionMean, 3);
  document.querySelector("#continuous-std").textContent = actionStandardDeviation.toFixed(3);
  document.querySelector("#continuous-value").textContent = predictedValue.toFixed(2);
  document.querySelector("#continuous-return").textContent = realizedReturn.toFixed(2);
  const advantageElement = document.querySelector("#continuous-advantage");
  advantageElement.textContent = formatSigned(advantage, 2);
  advantageElement.parentElement.classList.toggle("positive", advantage >= 0);
  advantageElement.parentElement.classList.toggle("negative", advantage < 0);
  document.querySelector("#continuous-position").textContent = formatSigned(position, 3);
  document.querySelector("#continuous-angle").textContent = `${formatSigned(angle, 3)} rad`;
  document.querySelector("#continuous-velocity").textContent = formatSigned(velocity, 3);
  document.querySelector("#continuous-angular-velocity").textContent = formatSigned(angularVelocity, 3);
  document.querySelector("#continuous-progress").style.width = `${(frameIndex + 1) / evaluation.trajectory.length * 100}%`;

  const actionRange = actionHigh - actionLow;
  const actionPercent = clampPercent((action - actionLow) / actionRange * 100);
  const bandStart = clampPercent(
    (actionMean - actionStandardDeviation - actionLow) / actionRange * 100
  );
  const bandEnd = clampPercent(
    (actionMean + actionStandardDeviation - actionLow) / actionRange * 100
  );
  document.querySelector("#continuous-force-marker").style.left = `${actionPercent}%`;
  const band = document.querySelector("#continuous-std-band");
  band.style.left = `${bandStart}%`;
  band.style.width = `${Math.max(1, bandEnd - bandStart)}%`;
}

function paintContinuousPendulum(canvas, frame, action, actionLow, actionHigh) {
  const [position, angle] = frame;
  const context = canvas.getContext("2d");
  fitCanvas(canvas);
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  context.clearRect(0, 0, width, height);

  const groundY = height * 0.78;
  context.strokeStyle = "#7c8d86";
  context.lineWidth = 3;
  context.beginPath();
  context.moveTo(35, groundY);
  context.lineTo(width - 35, groundY);
  context.stroke();

  const centerX = width / 2 + (position / 2.4) * width * 0.38;
  const cartWidth = Math.max(76, width * 0.14);
  const cartHeight = 36;
  context.fillStyle = "#2f7693";
  roundedRect(context, centerX - cartWidth / 2, groundY - cartHeight, cartWidth, cartHeight, 8);
  context.fill();

  context.fillStyle = "#17231f";
  context.beginPath();
  context.arc(centerX - cartWidth * 0.3, groundY + 6, 10, 0, Math.PI * 2);
  context.arc(centerX + cartWidth * 0.3, groundY + 6, 10, 0, Math.PI * 2);
  context.fill();

  const pivotY = groundY - cartHeight;
  const poleLength = Math.min(165, height * 0.48);
  const poleEndX = centerX + Math.sin(angle) * poleLength;
  const poleEndY = pivotY - Math.cos(angle) * poleLength;
  context.strokeStyle = Math.abs(angle) > 0.15 ? "#d85c52" : "#f2bd4b";
  context.lineWidth = 10;
  context.lineCap = "round";
  context.beginPath();
  context.moveTo(centerX, pivotY);
  context.lineTo(poleEndX, poleEndY);
  context.stroke();
  context.fillStyle = "#17231f";
  context.beginPath();
  context.arc(centerX, pivotY, 8, 0, Math.PI * 2);
  context.fill();

  // 箭头长度直接对应推力大小，让“轻推”和“重推”的差别一眼可见。
  const normalizedAction = action / Math.max(Math.abs(actionLow), Math.abs(actionHigh));
  if (Math.abs(normalizedAction) > 0.01) {
    const arrowY = groundY - cartHeight / 2;
    const arrowLength = normalizedAction * width * 0.13;
    const arrowEndX = centerX + arrowLength;
    const direction = Math.sign(arrowLength);
    context.strokeStyle = direction > 0 ? "#28ad7c" : "#d85c52";
    context.fillStyle = context.strokeStyle;
    context.lineWidth = 5;
    context.beginPath();
    context.moveTo(centerX, arrowY);
    context.lineTo(arrowEndX, arrowY);
    context.stroke();
    context.beginPath();
    context.moveTo(arrowEndX, arrowY);
    context.lineTo(arrowEndX - direction * 12, arrowY - 8);
    context.lineTo(arrowEndX - direction * 12, arrowY + 8);
    context.closePath();
    context.fill();
  }

  context.fillStyle = "#64706a";
  context.font = "13px sans-serif";
  context.textAlign = "left";
  context.fillText(`连续推力 ${formatSigned(action, 3)}`, 24, 28);
  context.fillText(`杆角度 ${formatSigned(angle, 3)} rad`, 24, 48);
}

function drawContinuousGaussian(canvas, mean, standardDeviation, action, low, high) {
  fitCanvas(canvas);
  const context = canvas.getContext("2d");
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const padding = { top: 24, right: 24, bottom: 40, left: 42 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const safeStandardDeviation = Math.max(0.05, standardDeviation);
  const samples = Array.from({ length: 121 }, (_, index) => {
    const x = low + index / 120 * (high - low);
    const z = (x - mean) / safeStandardDeviation;
    return { x, density: Math.exp(-0.5 * z * z) };
  });
  const maxDensity = Math.max(...samples.map((sample) => sample.density));
  const toCanvasX = (value) => padding.left + (value - low) / (high - low) * plotWidth;

  context.clearRect(0, 0, width, height);
  context.strokeStyle = "#d9ddd6";
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(padding.left, padding.top + plotHeight);
  context.lineTo(width - padding.right, padding.top + plotHeight);
  context.stroke();

  // 为了让初学者直观看宽窄，这里画动作尺度上的近似钟形曲线，而不是展开变换后的完整密度公式。
  context.beginPath();
  samples.forEach((sample, index) => {
    const x = toCanvasX(sample.x);
    const y = padding.top + plotHeight * (1 - sample.density / maxDensity * 0.9);
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.lineTo(width - padding.right, padding.top + plotHeight);
  context.lineTo(padding.left, padding.top + plotHeight);
  context.closePath();
  context.fillStyle = "rgba(40,173,124,.18)";
  context.fill();
  context.strokeStyle = "#136f55";
  context.lineWidth = 3;
  context.stroke();

  const meanX = toCanvasX(mean);
  const actionX = toCanvasX(action);
  context.setLineDash([5, 4]);
  context.strokeStyle = "#136f55";
  context.beginPath();
  context.moveTo(meanX, padding.top);
  context.lineTo(meanX, padding.top + plotHeight);
  context.stroke();
  context.setLineDash([]);
  context.fillStyle = "#16324f";
  context.fillRect(actionX - 2, padding.top + plotHeight - 22, 4, 22);

  context.fillStyle = "#64706a";
  context.font = "11px sans-serif";
  context.textAlign = "center";
  [low, 0, high].forEach((value) => {
    context.fillText(formatSigned(value, 0), toCanvasX(value), height - 12);
  });
  context.textAlign = "left";
  context.fillText(`中心=${formatSigned(mean, 2)}，近似宽度=${standardDeviation.toFixed(2)}`, padding.left, 14);
}

function renderContinuousMetrics() {
  const data = state.continuousData;
  document.querySelector("#continuous-metrics").innerHTML = [
    `<span>实际 ${data.steps_completed.toLocaleString()} 步</span>`,
    `<span>更新 ${data.update_count} 次</span>`,
    `<span>平均评估 ${data.evaluation.average_return.toFixed(1)}</span>`,
    `<span>参数 ${data.parameter_count.toLocaleString()}</span>`,
  ].join("");
}

function renderContinuousDiagnostics() {
  const data = state.continuousData;
  const averageClipFraction = data.clip_fraction_history.reduce(
    (sum, value) => sum + value,
    0
  ) / Math.max(1, data.clip_fraction_history.length);
  const finalStandardDeviation = data.action_std_history.at(-1) ?? 0;
  const finalKL = data.approximate_kl_history.at(-1) ?? 0;
  document.querySelector("#continuous-range").textContent = `${data.action_low[0]} ～ +${data.action_high[0]}`;
  document.querySelector("#continuous-steps").textContent = data.steps_completed.toLocaleString();
  document.querySelector("#continuous-final-std").textContent = finalStandardDeviation.toFixed(3);
  document.querySelector("#continuous-clip-fraction").textContent = `${(averageClipFraction * 100).toFixed(1)}%`;
  document.querySelector("#continuous-kl").textContent = finalKL.toFixed(4);
}

async function loadSAC() {
  const status = document.querySelector("#sac-status");
  try {
    state.sacData = await loadJson("data/sac.json");
    status.className = "data-status ready";
    status.textContent = "训练数据读取成功。动画展示接近平均水平的一条轨迹，而不是挑选开局最幸运的一局。";
    renderSACMetrics();
    renderSACDiagnostics();
    drawSACRewardChart();
    startSACPlayback();
  } catch (error) {
    status.className = "data-status error";
    status.textContent = `没有找到 SAC 数据。请运行 ./scripts/run_python.sh scripts/train_sac.py --steps 100000。错误：${error.message}`;
    drawCanvasMessage(
      document.querySelector("#sac-canvas"),
      "完成 SAC 训练后，这里会显示摆锤动画。"
    );
  }
}

function startSACPlayback() {
  if (!state.sacData || state.sacTimer) return;
  drawSACFrame();
  state.sacTimer = window.setInterval(() => {
    if (!state.sacPlaying) return;
    const trajectory = state.sacData.evaluation.trajectory;
    state.sacFrame = (state.sacFrame + 1) % trajectory.length;
    drawSACFrame();
  }, 45);
}

function stopSACPlayback() {
  if (state.sacTimer) window.clearInterval(state.sacTimer);
  state.sacTimer = null;
}

function drawSACFrame() {
  if (!state.sacData) return;
  const data = state.sacData;
  const evaluation = data.evaluation;
  const frameIndex = Math.min(state.sacFrame, evaluation.trajectory.length - 1);
  const [cosine, sine, angularVelocity] = evaluation.trajectory[frameIndex];
  const angle = Math.atan2(sine, cosine);
  const torque = evaluation.actions[frameIndex][0];
  const actionCenter = evaluation.action_centers[frameIndex][0];
  const actionStandardDeviation = evaluation.action_standard_deviations[frameIndex][0];
  const reward = evaluation.rewards[frameIndex];
  const qValueOne = evaluation.q_values_one[frameIndex];
  const qValueTwo = evaluation.q_values_two[frameIndex];
  const realizedReturn = evaluation.realized_returns[frameIndex];
  const actionLow = data.action_low[0];
  const actionHigh = data.action_high[0];

  paintSACPendulum(
    document.querySelector("#sac-canvas"),
    angle,
    angularVelocity,
    torque,
    actionLow,
    actionHigh
  );

  document.querySelector("#sac-torque").textContent = formatSigned(torque, 3);
  document.querySelector("#sac-center").textContent = formatSigned(actionCenter, 3);
  document.querySelector("#sac-std").textContent = actionStandardDeviation.toFixed(3);
  document.querySelector("#sac-reward").textContent = reward.toFixed(2);
  document.querySelector("#sac-q-one").textContent = qValueOne.toFixed(2);
  document.querySelector("#sac-q-two").textContent = qValueTwo.toFixed(2);
  document.querySelector("#sac-q-min").textContent = Math.min(qValueOne, qValueTwo).toFixed(2);
  document.querySelector("#sac-return").textContent = realizedReturn.toFixed(2);
  document.querySelector("#sac-angle").textContent = `${formatSigned(angle, 3)} rad`;
  document.querySelector("#sac-angular-velocity").textContent = formatSigned(angularVelocity, 3);
  document.querySelector("#sac-progress").style.width = `${(frameIndex + 1) / evaluation.trajectory.length * 100}%`;

  const actionRange = actionHigh - actionLow;
  const actionPercent = clampPercent((torque - actionLow) / actionRange * 100);
  const bandStart = clampPercent(
    (actionCenter - actionStandardDeviation - actionLow) / actionRange * 100
  );
  const bandEnd = clampPercent(
    (actionCenter + actionStandardDeviation - actionLow) / actionRange * 100
  );
  document.querySelector("#sac-torque-marker").style.left = `${actionPercent}%`;
  const band = document.querySelector("#sac-std-band");
  band.style.left = `${bandStart}%`;
  band.style.width = `${Math.max(1, bandEnd - bandStart)}%`;
}

function paintSACPendulum(canvas, angle, angularVelocity, torque, actionLow, actionHigh) {
  const context = canvas.getContext("2d");
  fitCanvas(canvas);
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  context.clearRect(0, 0, width, height);

  const pivotX = width / 2;
  const pivotY = height * 0.54;
  const pendulumLength = Math.min(165, height * 0.38);
  const endX = pivotX + Math.sin(angle) * pendulumLength;
  const endY = pivotY - Math.cos(angle) * pendulumLength;

  context.setLineDash([6, 6]);
  context.strokeStyle = "rgba(19,111,85,.28)";
  context.lineWidth = 2;
  context.beginPath();
  context.moveTo(pivotX, pivotY);
  context.lineTo(pivotX, pivotY - pendulumLength - 28);
  context.stroke();
  context.setLineDash([]);

  context.strokeStyle = "#16324f";
  context.lineWidth = 12;
  context.lineCap = "round";
  context.beginPath();
  context.moveTo(pivotX, pivotY);
  context.lineTo(endX, endY);
  context.stroke();

  context.fillStyle = Math.abs(angle) < 0.2 ? "#28ad7c" : "#f2bd4b";
  context.beginPath();
  context.arc(endX, endY, 24, 0, Math.PI * 2);
  context.fill();
  context.fillStyle = "#17231f";
  context.beginPath();
  context.arc(pivotX, pivotY, 12, 0, Math.PI * 2);
  context.fill();

  // 用绕轴圆弧表示转矩方向，比直线箭头更接近“拧动摆锤”的实际含义。
  const normalizedTorque = torque / Math.max(Math.abs(actionLow), Math.abs(actionHigh));
  if (Math.abs(normalizedTorque) > 0.02) {
    const clockwise = normalizedTorque > 0;
    const radius = 55;
    const startAngle = clockwise ? -Math.PI * 0.85 : -Math.PI * 0.15;
    const endAngle = clockwise ? Math.PI * 0.35 : -Math.PI * 1.35;
    context.strokeStyle = clockwise ? "#28ad7c" : "#d85c52";
    context.fillStyle = context.strokeStyle;
    context.lineWidth = 5;
    context.beginPath();
    context.arc(pivotX, pivotY, radius, startAngle, endAngle, !clockwise);
    context.stroke();
    const arrowX = pivotX + Math.cos(endAngle) * radius;
    const arrowY = pivotY + Math.sin(endAngle) * radius;
    context.beginPath();
    context.arc(arrowX, arrowY, 7, 0, Math.PI * 2);
    context.fill();
  }

  context.fillStyle = "#64706a";
  context.font = "13px sans-serif";
  context.textAlign = "left";
  context.fillText(`转矩 ${formatSigned(torque, 3)}`, 24, 28);
  context.fillText(`角度 ${formatSigned(angle, 3)} rad`, 24, 48);
  context.fillText(`角速度 ${formatSigned(angularVelocity, 3)}`, 24, 68);
}

function renderSACMetrics() {
  const data = state.sacData;
  document.querySelector("#sac-metrics").innerHTML = [
    `<span>实际 ${data.steps_completed.toLocaleString()} 步</span>`,
    `<span>平均评估 ${data.evaluation.average_return.toFixed(1)}</span>`,
    `<span>代表轨迹 ${data.evaluation.trajectory_return.toFixed(1)}</span>`,
    `<span>最佳单局 ${data.evaluation.best_return.toFixed(1)}</span>`,
  ].join("");
}

function renderSACDiagnostics() {
  const data = state.sacData;
  const finalQDisagreement = data.q_disagreement_history.at(-1) ?? 0;
  const finalReplaySize = data.replay_size_history.at(-1) ?? data.steps_completed;
  const replayPercent = Math.min(
    100,
    finalReplaySize / data.config.replay_capacity * 100
  );
  document.querySelector("#sac-steps").textContent = data.steps_completed.toLocaleString();
  document.querySelector("#sac-updates").textContent = data.update_count.toLocaleString();
  document.querySelector("#sac-final-temperature").textContent = data.final_temperature.toFixed(3);
  document.querySelector("#sac-temperature").textContent = data.final_temperature.toFixed(3);
  document.querySelector("#sac-q-disagreement").textContent = finalQDisagreement.toFixed(3);
  document.querySelector("#sac-parameters").textContent = data.parameter_count.toLocaleString();
  document.querySelector("#sac-trajectory-return").textContent = data.evaluation.trajectory_return.toFixed(1);
  document.querySelector("#sac-buffer-size").textContent = `${finalReplaySize.toLocaleString()} / ${data.config.replay_capacity.toLocaleString()}`;
  document.querySelector("#sac-buffer-fill").style.width = `${replayPercent}%`;
}

function formatSigned(value, digits) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;
}

function clampPercent(value) {
  return Math.min(100, Math.max(0, value));
}

function drawGridRewardChart() {
  const values = state.gridData.training.episode_rewards;
  drawLineChart(document.querySelector("#grid-reward-chart"), values, {
    averageWindow: 50,
    minY: Math.min(-11, ...values),
    maxY: Math.max(10, ...values),
  });
}

function drawCartPoleRewardChart() {
  const values = state.cartpoleData.episode_returns;
  drawLineChart(document.querySelector("#cartpole-reward-chart"), values, {
    averageWindow: 20,
    minY: 0,
    maxY: Math.max(500, ...values),
  });
}

function drawPolicyRewardChart() {
  const values = state.policyData.episode_returns;
  drawLineChart(document.querySelector("#policy-reward-chart"), values, {
    averageWindow: 25,
    minY: 0,
    maxY: Math.max(500, ...values),
  });
}

function drawActorCriticRewardChart() {
  const values = state.actorCriticData.episode_returns;
  drawLineChart(document.querySelector("#actor-critic-reward-chart"), values, {
    averageWindow: 50,
    minY: 0,
    maxY: Math.max(500, ...values),
  });
}

function drawPPORewardChart() {
  const values = state.ppoData.episode_returns;
  drawLineChart(document.querySelector("#ppo-reward-chart"), values, {
    averageWindow: 50,
    minY: 0,
    maxY: Math.max(500, ...values),
  });
}

function drawContinuousRewardChart() {
  const values = state.continuousData.evaluation_history.map(
    (checkpoint) => checkpoint.average_return
  );
  drawLineChart(document.querySelector("#continuous-reward-chart"), values, {
    averageWindow: 1,
    minY: 0,
    maxY: 1000,
  });
}

function drawSACRewardChart() {
  const values = state.sacData.evaluation_history.map(
    (checkpoint) => checkpoint.average_return
  );
  drawLineChart(document.querySelector("#sac-reward-chart"), values, {
    averageWindow: 1,
    minY: Math.min(-1000, ...values),
    maxY: 0,
  });
}

function drawLineChart(canvas, values, options) {
  if (!values?.length) return;
  fitCanvas(canvas);
  const context = canvas.getContext("2d");
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const padding = { top: 25, right: 20, bottom: 36, left: 48 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const minY = options.minY;
  const maxY = options.maxY;
  const average = movingAverage(values, options.averageWindow);
  context.clearRect(0, 0, width, height);

  context.strokeStyle = "#dce1dd";
  context.lineWidth = 1;
  context.fillStyle = "#75817c";
  context.font = "11px sans-serif";
  for (let index = 0; index <= 4; index += 1) {
    const y = padding.top + plotHeight * index / 4;
    const value = maxY - (maxY - minY) * index / 4;
    context.beginPath();
    context.moveTo(padding.left, y);
    context.lineTo(width - padding.right, y);
    context.stroke();
    context.fillText(value.toFixed(0), 6, y + 4);
  }

  plotSeries(context, values, "rgba(19,111,85,.18)", 1, padding, plotWidth, plotHeight, minY, maxY);
  plotSeries(context, average, "#136f55", 3, padding, plotWidth, plotHeight, minY, maxY);
  context.fillStyle = "#75817c";
  context.textAlign = "center";
  context.fillText("训练进度 →", padding.left + plotWidth / 2, height - 8);
  context.textAlign = "left";
}

function plotSeries(context, values, color, lineWidth, padding, plotWidth, plotHeight, minY, maxY) {
  context.strokeStyle = color;
  context.lineWidth = lineWidth;
  context.beginPath();
  values.forEach((value, index) => {
    const x = padding.left + index / Math.max(1, values.length - 1) * plotWidth;
    const normalized = (value - minY) / Math.max(0.0001, maxY - minY);
    const y = padding.top + plotHeight * (1 - normalized);
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.stroke();
}

function movingAverage(values, windowSize) {
  let sum = 0;
  return values.map((value, index) => {
    sum += value;
    if (index >= windowSize) sum -= values[index - windowSize];
    return sum / Math.min(index + 1, windowSize);
  });
}

function redrawCharts() {
  if (state.gridData) drawGridRewardChart();
  if (state.cartpoleData) {
    drawCartPoleRewardChart();
    drawCartPoleFrame();
  } else drawCartPolePlaceholder();
  if (state.policyData) {
    drawPolicyRewardChart();
    drawPolicyFrame();
  } else {
    drawCanvasMessage(
      document.querySelector("#policy-canvas"),
      "完成策略梯度训练后，这里会显示动作概率动画。"
    );
  }
  if (state.actorCriticData) {
    drawActorCriticRewardChart();
    drawActorCriticFrame();
  } else {
    drawCanvasMessage(
      document.querySelector("#actor-critic-canvas"),
      "完成行动者-评论家训练后，这里会显示双角色动画。"
    );
  }
  if (state.ppoData) {
    drawPPORewardChart();
    drawPPOFrame();
  } else {
    drawCanvasMessage(
      document.querySelector("#ppo-canvas"),
      "完成 PPO 训练后，这里会显示策略限速实验。"
    );
  }
  if (state.continuousData) {
    drawContinuousRewardChart();
    drawContinuousFrame();
  } else {
    drawCanvasMessage(
      document.querySelector("#continuous-canvas"),
      "完成 MuJoCo 连续动作训练后，这里会显示倒立摆动画。"
    );
    drawCanvasMessage(
      document.querySelector("#continuous-gaussian-canvas"),
      "完成训练后，这里会显示动作分布。"
    );
  }
  if (state.sacData) {
    drawSACRewardChart();
    drawSACFrame();
  } else {
    drawCanvasMessage(
      document.querySelector("#sac-canvas"),
      "完成 SAC 训练后，这里会显示摆锤动画。"
    );
  }
}

function fitCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || canvas.parentElement.clientWidth;
  // 使用 CSS 布局高度而不是读取会被像素密度放大的内部画布高度，
  // 否则高分屏每次重绘都可能再次把画布高度乘以设备像素比。
  const height = canvas.clientHeight || 300;
  if (canvas.width !== width * ratio || canvas.height !== height * ratio) {
    canvas.width = width * ratio;
    canvas.height = height * ratio;
  }
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  canvas.style.height = `${height}px`;
}

function roundedRect(context, x, y, width, height, radius) {
  context.beginPath();
  context.roundRect(x, y, width, height, radius);
}

function positionToState(position) {
  return position[0] * state.gridData.environment.cols + position[1];
}

function stateToPosition(index, cols) {
  return [Math.floor(index / cols), index % cols];
}

function equalPosition(first, second) {
  return first[0] === second[0] && first[1] === second[1];
}

function samePosition(position, positions) {
  return positions.some((item) => equalPosition(position, item));
}
