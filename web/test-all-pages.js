const { chromium } = require('playwright');

const pages = [
  {
    url: 'http://localhost:3000/characters',
    name: 'character-list',
    description: 'Character list (should show 6 character cards with gradients)'
  },
  {
    url: 'http://localhost:3000/characters/chr_1',
    name: 'character-detail',
    description: 'Character detail (should show cover left, info right, 角色介绍 + 开场白 sections)'
  },
  {
    url: 'http://localhost:3000/chat/ses_1',
    name: 'chat-page',
    description: 'Chat page (should show session list sidebar + chat messages)'
  },
  {
    url: 'http://localhost:3000/worldbooks',
    name: 'worldbook-list',
    description: 'World book list (should show 5 worldbooks with scope badges and filter tabs)'
  },
  {
    url: 'http://localhost:3000/profile',
    name: 'profile-page',
    description: 'Profile page (should show user info + stats + chart)'
  }
];

(async () => {
  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: 1920, height: 1080 }
  });
  const page = await context.newPage();

  console.log('开始测试页面...\n');

  for (const pageInfo of pages) {
    console.log(`\n========================================`);
    console.log(`测试: ${pageInfo.name}`);
    console.log(`URL: ${pageInfo.url}`);
    console.log(`描述: ${pageInfo.description}`);
    console.log(`========================================`);

    try {
      // 导航到页面
      const response = await page.goto(pageInfo.url, {
        waitUntil: 'networkidle',
        timeout: 10000
      });

      console.log(`✓ 页面加载成功 (状态码: ${response.status()})`);

      // 等待页面渲染
      await page.waitForTimeout(2000);

      // 检查是否有错误信息
      const errorElements = await page.locator('text=/error|错误|failed|失败/i').count();
      if (errorElements > 0) {
        console.log(`⚠ 警告: 页面上检测到 ${errorElements} 个可能的错误信息`);
      }

      // 检查页面是否为空
      const bodyText = await page.locator('body').textContent();
      if (bodyText.trim().length < 10) {
        console.log(`⚠ 警告: 页面内容似乎为空`);
      }

      // 截图
      const screenshotPath = `screenshots/${pageInfo.name}.png`;
      await page.screenshot({
        path: screenshotPath,
        fullPage: true
      });
      console.log(`✓ 截图已保存: ${screenshotPath}`);

      // 获取页面标题
      const title = await page.title();
      console.log(`✓ 页面标题: ${title}`);

      // 检查控制台错误
      const consoleErrors = [];
      page.on('console', msg => {
        if (msg.type() === 'error') {
          consoleErrors.push(msg.text());
        }
      });

      // 检查页面错误
      const pageErrors = [];
      page.on('pageerror', error => {
        pageErrors.push(error.message);
      });

      if (consoleErrors.length > 0) {
        console.log(`⚠ 控制台错误 (${consoleErrors.length}):`);
        consoleErrors.forEach(err => console.log(`  - ${err}`));
      }

      if (pageErrors.length > 0) {
        console.log(`⚠ 页面错误 (${pageErrors.length}):`);
        pageErrors.forEach(err => console.log(`  - ${err}`));
      }

      if (consoleErrors.length === 0 && pageErrors.length === 0) {
        console.log(`✓ 无控制台或页面错误`);
      }

    } catch (error) {
      console.log(`✗ 错误: ${error.message}`);
    }
  }

  console.log('\n\n========================================');
  console.log('所有页面测试完成！');
  console.log('========================================');

  await browser.close();
})();
