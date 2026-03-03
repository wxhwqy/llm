const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

(async () => {
  const screenshotsDir = path.join(__dirname, 'screenshots');
  
  if (!fs.existsSync(screenshotsDir)) {
    fs.mkdirSync(screenshotsDir, { recursive: true });
  }

  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: 1920, height: 1080 }
  });
  const page = await context.newPage();

  // Page 1: Characters page (dark theme)
  console.log('\n=== 1. 角色卡页面 (暗色主题) ===');
  console.log('URL: http://localhost:3000/characters');
  await page.goto('http://localhost:3000/characters', { waitUntil: 'networkidle', timeout: 15000 });
  await page.waitForTimeout(3000);
  await page.screenshot({ path: path.join(screenshotsDir, '1-characters-dark.png'), fullPage: true });
  console.log('✓ 截图已保存: 1-characters-dark.png');

  // Page 2: Character edit page
  console.log('\n=== 2. 角色编辑页面 ===');
  console.log('URL: http://localhost:3000/characters/c1/edit');
  await page.goto('http://localhost:3000/characters/c1/edit', { waitUntil: 'networkidle', timeout: 15000 });
  await page.waitForTimeout(3000);
  await page.screenshot({ path: path.join(screenshotsDir, '2-character-edit.png'), fullPage: true });
  console.log('✓ 截图已保存: 2-character-edit.png');

  // Page 3: Chat page
  console.log('\n=== 3. 聊天页面 ===');
  console.log('URL: http://localhost:3000/chat/s1');
  await page.goto('http://localhost:3000/chat/s1', { waitUntil: 'networkidle', timeout: 15000 });
  await page.waitForTimeout(3000);
  await page.screenshot({ path: path.join(screenshotsDir, '3-chat-s1.png'), fullPage: true });
  console.log('✓ 截图已保存: 3-chat-s1.png');

  // Page 4: Characters page with theme toggle
  console.log('\n=== 4. 角色卡页面 (切换到亮色主题) ===');
  console.log('URL: http://localhost:3000/characters');
  await page.goto('http://localhost:3000/characters', { waitUntil: 'networkidle', timeout: 15000 });
  await page.waitForTimeout(2000);
  
  console.log('正在查找主题切换按钮...');
  const selectors = [
    'button[aria-label*="theme"]',
    'button[aria-label*="主题"]',
    '[data-theme-toggle]',
    'header button:has(svg)',
    'button svg'
  ];
  
  let clicked = false;
  for (const selector of selectors) {
    try {
      const buttons = await page.$$(selector);
      if (buttons.length > 0) {
        await buttons[0].click();
        console.log(`✓ 点击了主题切换按钮 (选择器: ${selector})`);
        clicked = true;
        await page.waitForTimeout(1000);
        break;
      }
    } catch (e) {
      // 继续尝试
    }
  }
  
  if (!clicked) {
    console.log('⚠ 未找到主题切换按钮,尝试手动查找...');
    const html = await page.content();
    await fs.promises.writeFile(path.join(screenshotsDir, 'debug-page-html.html'), html);
  }
  
  await page.screenshot({ path: path.join(screenshotsDir, '4-characters-light.png'), fullPage: true });
  console.log('✓ 截图已保存: 4-characters-light.png');

  await browser.close();
  console.log('\n所有截图已完成!');
  console.log(`截图保存在: ${screenshotsDir}`);
})();
