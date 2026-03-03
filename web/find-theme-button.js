const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

(async () => {
  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: 1920, height: 1080 }
  });
  const page = await context.newPage();

  await page.goto('http://localhost:3000/characters', { waitUntil: 'networkidle' });
  await page.waitForTimeout(2000);
  
  // 获取header区域的HTML
  const headerHTML = await page.$eval('header', el => el.outerHTML).catch(() => null);
  if (headerHTML) {
    fs.writeFileSync(path.join(__dirname, 'screenshots', 'header.html'), headerHTML);
    console.log('Header HTML已保存');
  }
  
  // 查找所有可能的主题切换元素
  const allButtons = await page.$$('button, [role="button"], a');
  console.log(`\n找到 ${allButtons.length} 个可点击元素\n`);
  
  for (let i = 0; i < Math.min(allButtons.length, 20); i++) {
    const el = allButtons[i];
    const tagName = await el.evaluate(e => e.tagName);
    const className = await el.getAttribute('class');
    const ariaLabel = await el.getAttribute('aria-label');
    const title = await el.getAttribute('title');
    const innerHTML = await el.innerHTML();
    
    console.log(`元素 ${i}:`);
    console.log(`  标签: ${tagName}`);
    console.log(`  类名: ${className}`);
    console.log(`  aria-label: ${ariaLabel}`);
    console.log(`  title: ${title}`);
    console.log(`  内容: ${innerHTML.substring(0, 100)}...`);
    console.log('---');
  }

  await browser.close();
})();
