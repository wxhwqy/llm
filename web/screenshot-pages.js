const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

const pages = [
  { url: 'http://localhost:3000/characters', name: '角色卡页面', filename: '1-characters.png' },
  { url: 'http://localhost:3000/chat/s1', name: '聊天页面', filename: '2-chat-s1.png' },
  { url: 'http://localhost:3000/worldbooks', name: '世界书列表页', filename: '3-worldbooks.png' },
  { url: 'http://localhost:3000/worldbooks/wb1', name: '世界书编辑页', filename: '4-worldbooks-wb1.png' },
  { url: 'http://localhost:3000/profile', name: '用户信息页', filename: '5-profile.png' },
  { url: 'http://localhost:3000/login', name: '登录页', filename: '6-login.png' },
];

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

  for (const pageInfo of pages) {
    console.log(`正在访问: ${pageInfo.name} (${pageInfo.url})`);
    
    try {
      await page.goto(pageInfo.url, { waitUntil: 'networkidle', timeout: 10000 });
      
      await page.waitForTimeout(2000);
      
      const screenshotPath = path.join(screenshotsDir, pageInfo.filename);
      await page.screenshot({ path: screenshotPath, fullPage: true });
      
      console.log(`✓ 截图已保存: ${pageInfo.filename}`);
    } catch (error) {
      console.error(`✗ 访问 ${pageInfo.name} 时出错:`, error.message);
    }
  }

  await browser.close();
  console.log('\n所有截图已完成!');
  console.log(`截图保存在: ${screenshotsDir}`);
})();
