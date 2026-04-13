const express = require("express");

const app = express();
const PORT = Number(process.env.PORT || 3100);

app.use(express.static("public"));

const server = app.listen(PORT, () => {
  console.log(`Dashboard running on http://localhost:${PORT}`);
});

server.on("error", (err) => {
  if (err && err.code === "EADDRINUSE") {
    console.log(
      `Port ${PORT} is already in use. Dashboard may already be running. Use \"lsof -nP -iTCP:${PORT} -sTCP:LISTEN\" to check.`
    );
    process.exit(0);
  }

  console.error("Dashboard failed to start:", err);
  process.exit(1);
});
