// Opens the live site with a fake microphone that plays a WAV, waits, prints the transcript.
// node scripts/live_check.js <url> <wav> <out.png> [wait-seconds]
const puppeteer = require(process.env.PUPPETEER_PATH ||
  "/Users/okba/Documents/portflio/job-search/outreach/node_modules/puppeteer-core");
(async () => {
  const [url, wav, out, wait = "110"] = process.argv.slice(2);
  const browser = await puppeteer.launch({
    executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    headless: "new",
    args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
      `--use-file-for-fake-audio-capture=${wav}%noloop`, "--no-sandbox", "--autoplay-policy=no-user-gesture-required"],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 900, height: 1000, deviceScaleFactor: 2 });
  await page.goto(url, { waitUntil: "networkidle0" });
  await page.click("#connect");
  await new Promise((r) => setTimeout(r, Number(wait) * 1000));
  console.log(await page.$eval("#transcript", (e) => e.innerText));
  console.log(await page.$eval("#latency", (e) => e.innerText));
  await page.screenshot({ path: out });
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
