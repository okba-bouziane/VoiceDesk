// Renders a text file as a terminal-style screenshot: node scripts/text_screenshot.js in.txt out.png "title"
const fs = require("fs");
const puppeteer = require(process.env.PUPPETEER_PATH ||
  "/Users/okba/Documents/portflio/job-search/outreach/node_modules/puppeteer-core");
(async () => {
  const [input, out, title = ""] = process.argv.slice(2);
  const text = fs.readFileSync(input, "utf8").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  const browser = await puppeteer.launch({
    executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", headless: "new",
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1100, height: 600, deviceScaleFactor: 2 });
  await page.setContent(`<body style="margin:0;background:#0f1613;color:#d6e4dc;font:14px/1.5 ui-monospace,Menlo,monospace">
    <div style="padding:10px 18px;background:#17211c;color:#93a39b;border-bottom:1px solid #27342d">${title}</div>
    <pre style="margin:0;padding:16px 18px;white-space:pre-wrap">${text}</pre></body>`);
  await page.screenshot({ path: out, fullPage: true });
  await browser.close();
})();
