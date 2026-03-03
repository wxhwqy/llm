const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

(async () => {
  const screenshotsDir = path.join(__dirname, 'screenshots');
  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: 1920, height: 1080 }
  });
  const page = await context.newPage();

  console.log('访问角色页面...');
  await page.goto('http://localhost:3000/characters', { waitUntil: 'networkidle', timeout: 15000 });
  await page.waitForTimeout(2000);
  
  console.log('获取页面HTML以查找主题切换按钮...');
  const buttons = await page.$$('button');
  console.log(`找到 ${buttons.length} 个按钮`);
  
  for (let i = 0; i < buttons.length; i++) {
    const button = buttons[i];
    const ariaLabel = await button.getAttribute('aria-label');
    const text = await button.textContent();
    const html = await button.innerHTML();
    
    if (ariaLabel || text || html.includes('svg')) {
      console.log(`按钮 ${i}: aria-label="${ariaLabel}", text="${text}", has-svg=${html.includes('svg')}`);
      
      if (ariaLabel && (ariaLabel.includes('theme') || ariaLabel.includes('主题') || ariaLabel.includes('Theme'))) {
        console.log(`✓ 找到主题切换按钮! 点击中...`);
        await button.click();
        await page.waitForTimeout(1000);
        break;
      }
    }
  }
  
  await page.screenshot({ path: path.join(screenshotsDir, '4-characters-light.png'), fullPage: true });
  console.log('✓ 截图已保存: 4-characters-light.png');

  await browser.close();
})();
