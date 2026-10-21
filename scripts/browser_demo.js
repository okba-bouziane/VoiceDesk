// Drives the web client in headless Chrome: joins a room, sends typed messages and takes
// screenshots. Usage: node scripts/browser_demo.js <out-dir>
const puppeteer = require(process.env.PUPPETEER_PATH ||
  "/Users/okba/Documents/portflio/job-search/outreach/node_modules/puppeteer-core");
const out = process.argv[2] || "docs/screenshots";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  const browser = await puppeteer.launch({
    executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    headless: "new",
    args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream", "--no-sandbox"],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 900, height: 1000, deviceScaleFactor: 2 });
  page.on("console", (m) => console.log("[console]", m.text()));
  await page.goto("http://localhost:8000", { waitUntil: "networkidle0" });
  await page.click("#connect");
  await page.waitForFunction(() => document.querySelector("#state").dataset.state !== "offline", { timeout: 20000 });
  const say = async (text) => {
    await page.type("#text", text);
    await page.keyboard.press("Enter");
    await sleep(9000);
  };
  await sleep(6000);
  for (const line of (process.env.DEMO_LINES || "What are your opening hours?|I want to book a dentist next Sunday at 10 in the morning").split("|")) {
    await say(line);
  }
  console.log(await page.$eval("#transcript", (e) => e.innerText));
  console.log(await page.$eval("#latency", (e) => e.innerText));
  await page.screenshot({ path: `${out}/web-client.png` });
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
