const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1920, height: 1080 }
  });
  const page = await context.newPage();

  try {
    console.log('\n=== 1. 导航到角色列表页面 ===');
    await page.goto('http://localhost:3000/characters', { waitUntil: 'networkidle' });
    await page.waitForTimeout(1000);
    console.log('URL:', page.url());
    await page.screenshot({ path: 'screenshots/test-1-characters-list.png', fullPage: false });
    console.log('✓ 截图已保存: test-1-characters-list.png');

    console.log('\n=== 2. 点击第一个角色卡片(艾莉丝) ===');
    // 查找第一个角色卡片
    const firstCard = await page.locator('[data-character-id="c1"]').first();
    if (await firstCard.count() === 0) {
      // 如果没有找到data-character-id,尝试其他选择器
      const cardLink = await page.locator('a[href*="/characters/c1"]').first();
      if (await cardLink.count() > 0) {
        await cardLink.click();
      } else {
        // 尝试找到包含"艾莉丝"的卡片
        const cards = await page.locator('text=艾莉丝').first();
        await cards.click();
      }
    } else {
      await firstCard.click();
    }
    
    // 等待导航完成
    await page.waitForURL('**/characters/c1', { timeout: 5000 });
    await page.waitForTimeout(1000);
    
    console.log('导航后的URL:', page.url());
    
    // 检查是否是新页面(URL应该改变)
    const currentUrl = page.url();
    const isNewPage = currentUrl.includes('/characters/c1');
    console.log('是否导航到新页面:', isNewPage ? '是 ✓' : '否 ✗');

    console.log('\n=== 3. 截图角色详情页面(上半部分) ===');
    await page.screenshot({ path: 'screenshots/test-2-character-detail-top.png', fullPage: false });
    console.log('✓ 截图已保存: test-2-character-detail-top.png');

    console.log('\n=== 4. 检查页面布局 ===');
    
    // 检查封面图片
    const coverImage = await page.locator('img[alt*="封面"], img[alt*="cover"], img').first();
    const coverExists = await coverImage.count() > 0;
    console.log('封面图片存在:', coverExists ? '是 ✓' : '否 ✗');
    
    // 检查标题
    const title = await page.locator('h1, h2').first();
    const titleExists = await title.count() > 0;
    const titleText = titleExists ? await title.textContent() : '';
    console.log('标题存在:', titleExists ? `是 ✓ (${titleText})` : '否 ✗');
    
    // 检查"开始对话"按钮
    const startChatButton = await page.locator('button:has-text("开始对话"), a:has-text("开始对话")');
    const buttonExists = await startChatButton.count() > 0;
    console.log('开始对话按钮存在:', buttonExists ? '是 ✓' : '否 ✗');
    
    if (buttonExists) {
      const buttonClasses = await startChatButton.first().getAttribute('class');
      console.log('按钮样式类:', buttonClasses);
    }

    console.log('\n=== 5. 滚动到页面底部查看详细内容 ===');
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await page.waitForTimeout(1000);
    await page.screenshot({ path: 'screenshots/test-3-character-detail-bottom.png', fullPage: false });
    console.log('✓ 截图已保存: test-3-character-detail-bottom.png');

    console.log('\n=== 6. 检查详情区域的标题样式 ===');
    // 查找带有左边框的标题
    const sectionHeadings = await page.locator('h2, h3, [class*="border-l"]');
    const headingCount = await sectionHeadings.count();
    console.log('找到的区域标题数量:', headingCount);
    
    if (headingCount > 0) {
      for (let i = 0; i < Math.min(headingCount, 3); i++) {
        const heading = sectionHeadings.nth(i);
        const text = await heading.textContent();
        const classes = await heading.getAttribute('class');
        console.log(`  标题 ${i + 1}: ${text?.trim()}`);
        console.log(`  样式类: ${classes}`);
      }
    }

    console.log('\n=== 7. 生成完整页面截图 ===');
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.waitForTimeout(500);
    await page.screenshot({ path: 'screenshots/test-4-character-detail-full.png', fullPage: true });
    console.log('✓ 截图已保存: test-4-character-detail-full.png');

    console.log('\n=== 测试总结 ===');
    console.log('1. 点击卡片是否导航到新页面:', isNewPage ? '是 ✓' : '否 ✗');
    console.log('2. 布局是否正确(封面+信息):', (coverExists && titleExists) ? '是 ✓' : '否 ✗');
    console.log('3. "开始对话"按钮是否显著:', buttonExists ? '是 ✓' : '否 ✗');
    console.log('4. 详情区域标题数量:', headingCount);

  } catch (error) {
    console.error('测试过程中出错:', error);
    await page.screenshot({ path: 'screenshots/test-error.png' });
  } finally {
    await browser.close();
  }
})();
