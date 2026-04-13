const SERVER_URL = "http://localhost:4100";
const socket = io(SERVER_URL, {
  transports: ["websocket", "polling"],
});

const els = {
  serverStatus: document.getElementById("server-status"),
  trainStatus: document.getElementById("train-status"),
  episode: document.getElementById("episode-value"),
  reward: document.getElementById("reward-value"),
  success: document.getElementById("success-value"),
  collision: document.getElementById("collision-value"),
  steps: document.getElementById("steps-value"),
  event: document.getElementById("event-value"),
  canvas: document.getElementById("trajectory-canvas"),
};

const ctxReward = document.getElementById("reward-chart").getContext("2d");
const ctxTrend = document.getElementById("trend-chart").getContext("2d");

const rewardChart = new Chart(ctxReward, {
  type: "line",
  data: {
    labels: [],
    datasets: [
      {
        label: "Episode Reward",
        data: [],
        borderColor: "#d3572a",
        backgroundColor: "rgba(211, 87, 42, 0.2)",
        borderWidth: 2,
        tension: 0.2,
        pointRadius: 0,
      },
      {
        label: "Avg Reward (100)",
        data: [],
        borderColor: "#2c7a7b",
        borderWidth: 2,
        tension: 0.25,
        pointRadius: 0,
      },
    ],
  },
  options: {
    maintainAspectRatio: false,
    scales: {
      x: { ticks: { maxTicksLimit: 12 } },
      y: { beginAtZero: false },
    },
    plugins: {
      legend: { labels: { boxWidth: 14 } },
    },
  },
});

const trendChart = new Chart(ctxTrend, {
  type: "line",
  data: {
    labels: [],
    datasets: [
      {
        label: "Success Rate",
        data: [],
        borderColor: "#2c7a7b",
        tension: 0.2,
        pointRadius: 0,
        borderWidth: 2,
      },
      {
        label: "Collision Rate",
        data: [],
        borderColor: "#b23a48",
        tension: 0.2,
        pointRadius: 0,
        borderWidth: 2,
      },
    ],
  },
  options: {
    maintainAspectRatio: false,
    scales: {
      y: {
        min: 0,
        max: 1,
        ticks: {
          callback: (v) => `${Math.round(v * 100)}%`,
        },
      },
      x: { ticks: { maxTicksLimit: 12 } },
    },
  },
});

function setPanelHeights() {
  const panels = document.querySelectorAll(".chart-panel");
  panels.forEach((panel) => {
    panel.style.height = window.innerWidth < 720 ? "260px" : "300px";
  });
}

function safePercent(value) {
  return `${(Math.max(0, Math.min(1, Number(value || 0))) * 100).toFixed(1)}%`;
}

function updateOutcomeBadge(data) {
  let text = "Outcome: waiting";
  let type = "neutral";

  const reason = data.termination_reason || null;
  if (reason === "goal_reached" || data.success) {
    text = "Outcome: goal reached";
    type = "safe";
  } else if (reason === "wall_collision" || data.collision_type === "wall") {
    text = "Outcome: wall collision";
    type = "danger";
  } else if (reason === "tape_collision" || data.collision_type === "tape") {
    text = "Outcome: floor tape collision";
    type = "danger";
  } else if (reason === "max_steps") {
    text = "Outcome: timeout (max steps)";
    type = "warn";
  } else if (data.collision) {
    text = "Outcome: collision";
    type = "danger";
  }

  if (els.event) {
    els.event.textContent = text;
    els.event.className = `event-pill ${type}`;
  }
}

function updateMetrics(data) {
  els.episode.textContent = String(data.episode ?? 0);
  els.reward.textContent = Number(data.reward ?? 0).toFixed(2);
  els.success.textContent = safePercent(data.success_rate);
  els.collision.textContent = safePercent(data.collision_rate);
  els.steps.textContent = `Steps: ${data.steps ?? 0}`;
  updateOutcomeBadge(data);

  rewardChart.data.labels.push(data.episode);
  rewardChart.data.datasets[0].data.push(data.reward ?? 0);
  rewardChart.data.datasets[1].data.push(data.avg_reward_100 ?? 0);

  trendChart.data.labels.push(data.episode);
  trendChart.data.datasets[0].data.push(data.success_rate ?? 0);
  trendChart.data.datasets[1].data.push(data.collision_rate ?? 0);

  trimChart(rewardChart, 250);
  trimChart(trendChart, 250);

  rewardChart.update("none");
  trendChart.update("none");

  drawTrajectory(data);
  pulse(els.reward);
}

function trimChart(chart, maxPoints) {
  while (chart.data.labels.length > maxPoints) {
    chart.data.labels.shift();
    chart.data.datasets.forEach((dataset) => dataset.data.shift());
  }
}

function pulse(element) {
  element.classList.remove("flash");
  // Force reflow to replay animation.
  void element.offsetWidth;
  element.classList.add("flash");
}

function drawTapeCell(ctx, x, y, cell) {
  ctx.fillStyle = "#f2c84f";
  ctx.fillRect(x, y, cell, cell);

  ctx.strokeStyle = "#1f1f1f";
  ctx.lineWidth = Math.max(1, cell * 0.08);
  for (let off = -cell; off < cell * 2; off += Math.max(4, cell * 0.28)) {
    ctx.beginPath();
    ctx.moveTo(x + off, y);
    ctx.lineTo(x + off + cell, y + cell);
    ctx.stroke();
  }
}

function drawRobotIcon(ctx, px, py, size, heading) {
  const r = size * 0.38;
  const left = heading + (Math.PI * 3) / 4;
  const right = heading - (Math.PI * 3) / 4;

  ctx.fillStyle = "#1c3d5a";
  ctx.beginPath();
  ctx.moveTo(px + Math.cos(heading) * r, py + Math.sin(heading) * r);
  ctx.lineTo(px + Math.cos(left) * r, py + Math.sin(left) * r);
  ctx.lineTo(px + Math.cos(right) * r, py + Math.sin(right) * r);
  ctx.closePath();
  ctx.fill();
}

function drawCollisionMarker(ctx, px, py, size) {
  const r = size * 0.28;
  ctx.strokeStyle = "#b23a48";
  ctx.lineWidth = Math.max(2, size * 0.1);
  ctx.beginPath();
  ctx.moveTo(px - r, py - r);
  ctx.lineTo(px + r, py + r);
  ctx.moveTo(px + r, py - r);
  ctx.lineTo(px - r, py + r);
  ctx.stroke();
}

function drawTrajectory(data) {
  const ctx = els.canvas.getContext("2d");
  const size = Number(data.grid_size || 20);
  const width = els.canvas.width;
  const cell = width / size;

  ctx.clearRect(0, 0, width, width);
  ctx.fillStyle = "#fdf7e8";
  ctx.fillRect(0, 0, width, width);

  ctx.strokeStyle = "rgba(120, 104, 79, 0.2)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= size; i += 1) {
    const p = i * cell;
    ctx.beginPath();
    ctx.moveTo(p, 0);
    ctx.lineTo(p, width);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(0, p);
    ctx.lineTo(width, p);
    ctx.stroke();
  }

  ctx.strokeStyle = "#4d4d56";
  ctx.lineWidth = Math.max(6, cell * 0.36);
  ctx.strokeRect(ctx.lineWidth / 2, ctx.lineWidth / 2, width - ctx.lineWidth, width - ctx.lineWidth);

  const tapeZones = data.tape_zones || data.obstacles || [];
  tapeZones.forEach(([x, y]) => {
    drawTapeCell(ctx, x * cell, y * cell, cell);
  });

  const goal = data.goal || [];
  if (goal.length === 2) {
    const gx = goal[0] * cell;
    const gy = goal[1] * cell;
    ctx.fillStyle = "#2c7a7b";
    ctx.fillRect(gx, gy, cell, cell);
    ctx.strokeStyle = "#d8f2ea";
    ctx.lineWidth = Math.max(1, cell * 0.08);
    ctx.strokeRect(gx + cell * 0.2, gy + cell * 0.2, cell * 0.6, cell * 0.6);
  }

  const path = data.trajectory || [];
  if (path.length > 0) {
    ctx.strokeStyle = "#2b73c5";
    ctx.lineWidth = Math.max(2, cell * 0.18);
    ctx.beginPath();
    path.forEach(([x, y], idx) => {
      const px = x * cell + cell / 2;
      const py = y * cell + cell / 2;
      if (idx === 0) {
        ctx.moveTo(px, py);
      } else {
        ctx.lineTo(px, py);
      }
    });
    ctx.stroke();

    const [sx, sy] = path[0];
    const [ex, ey] = path[path.length - 1];

    ctx.fillStyle = "#205e4c";
    ctx.beginPath();
    ctx.arc(sx * cell + cell / 2, sy * cell + cell / 2, cell * 0.2, 0, Math.PI * 2);
    ctx.fill();

    const endPx = ex * cell + cell / 2;
    const endPy = ey * cell + cell / 2;
    let heading = -Math.PI / 2;
    if (path.length >= 2) {
      const [px0, py0] = path[path.length - 2];
      heading = Math.atan2(ey - py0, ex - px0);
    }
    drawRobotIcon(ctx, endPx, endPy, cell, heading);
  }

  if (data.collision) {
    let marker = null;
    if (Array.isArray(data.collision_point) && data.collision_point.length === 2) {
      marker = data.collision_point;
    } else if (path.length > 0) {
      marker = path[path.length - 1];
    }
    if (marker) {
      drawCollisionMarker(ctx, marker[0] * cell + cell / 2, marker[1] * cell + cell / 2, cell);
    }
  }
}

async function loadHistory() {
  try {
    const response = await fetch(`${SERVER_URL}/history`);
    const data = await response.json();

    if (data.status?.status) {
      els.trainStatus.textContent = `Training: ${data.status.status}`;
    }

    const metrics = Array.isArray(data.metrics) ? data.metrics : [];
    metrics.forEach((item) => updateMetrics(item));
  } catch (_err) {
    els.serverStatus.textContent = "Server: unavailable";
  }
}

socket.on("connect", () => {
  els.serverStatus.textContent = "Server: connected";
});

socket.on("disconnect", () => {
  els.serverStatus.textContent = "Server: disconnected";
});

socket.on("status", (status) => {
  els.trainStatus.textContent = `Training: ${status.status || "idle"}`;
});

socket.on("metrics", (payload) => {
  updateMetrics(payload);
});

setPanelHeights();
window.addEventListener("resize", setPanelHeights);
loadHistory();
