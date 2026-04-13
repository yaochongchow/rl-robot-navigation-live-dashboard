const express = require("express");
const http = require("http");
const cors = require("cors");
const { Server } = require("socket.io");

const PORT = Number(process.env.PORT || 4100);
const MAX_HISTORY = Number(process.env.MAX_HISTORY || 1000);

const app = express();
app.use(cors());
app.use(express.json({ limit: "1mb" }));

const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: "*",
  },
});

const metricsHistory = [];
let lastStatus = { status: "idle" };

app.get("/", (_req, res) => {
  res.json({
    service: "rl-nav-metrics-server",
    ok: true,
    endpoints: ["/health", "/history", "/metrics (POST)", "/status (POST)"],
  });
});

app.get("/health", (_req, res) => {
  res.json({ ok: true, status: lastStatus.status, historyCount: metricsHistory.length });
});

app.get("/history", (_req, res) => {
  res.json({ metrics: metricsHistory, status: lastStatus });
});

app.post("/metrics", (req, res) => {
  const payload = {
    ...req.body,
    timestamp: req.body?.timestamp || new Date().toISOString(),
  };

  metricsHistory.push(payload);
  if (metricsHistory.length > MAX_HISTORY) {
    metricsHistory.shift();
  }

  io.emit("metrics", payload);
  res.status(202).json({ accepted: true });
});

app.post("/status", (req, res) => {
  lastStatus = {
    ...req.body,
    timestamp: new Date().toISOString(),
  };

  io.emit("status", lastStatus);
  res.status(202).json({ accepted: true });
});

io.on("connection", (socket) => {
  socket.emit("status", lastStatus);
});

server.on("error", (err) => {
  if (err && err.code === "EADDRINUSE") {
    console.log(
      `Port ${PORT} is already in use. Metrics server may already be running. Use \"lsof -nP -iTCP:${PORT} -sTCP:LISTEN\" to check.`
    );
    process.exit(0);
  }

  console.error("Metrics server failed to start:", err);
  process.exit(1);
});

server.listen(PORT, () => {
  console.log(`Metrics server listening on http://localhost:${PORT}`);
});
