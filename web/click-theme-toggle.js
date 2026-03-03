const { chromium } = require('playwright');
const path = require('path');

(async () => {
  const screenshotsDir = path.join(__dirname, 'screenshots');
  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: 1920, height: 1080 }
  });
  const page = await context.newPage();

  console.log('访问角色页面...');
  await page.goto('http://localhost:3000/characters', { waitUntil: 'networkidle' });
  await page.waitForTimeout(2000);
  
  console.log('查找主题切换按钮 (moon/sun icon)...');
  
  // 查找包含 moon 或 sun 类的按钮
  const themeButton = await page.$('button:has(.lucide-moon), button:has(.lucide-sun)');
  
  if (themeButton) {
    console.log('✓ 找到主题切换按钮,点击中...');
    await themeButton.click();
    await page.waitForTimeout(1000);
    console.log('✓ 主题已切换');
  } else {
    console.log('✗ 未找到主题切换按钮');
  }
  
  await page.screenshot({ 
    path: path.join(screenshotsDir, '4-characters-light.png'), 
    fullPage: true 
  });
  console.log('✓ 截图已保存: 4-characters-light.png');

  await browser.close();
})();
