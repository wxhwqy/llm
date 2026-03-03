const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

async function takeScreenshots() {
  const browser = await puppeteer.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1920, height: 1080 });

    const screenshotsDir = path.join(__dirname, 'screenshots');
    if (!fs.existsSync(screenshotsDir)) {
      fs.mkdirSync(screenshotsDir, { recursive: true });
    }

    console.log('\n=== 1. 世界书列表页面 ===');
    console.log('URL: http://localhost:3000/worldbooks\n');
    
    await page.goto('http://localhost:3000/worldbooks', {
      waitUntil: 'networkidle0',
      timeout: 30000
    });
    
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // 截图1：全部世界书
    const screenshot1Path = path.join(screenshotsDir, 'worldbooks-all.png');
    await page.screenshot({ path: screenshot1Path, fullPage: true });
    console.log(`✓ 截图已保存: ${screenshot1Path}`);
    
    // 检查页面内容
    const pageContent = await page.evaluate(() => {
      const filterTabs = Array.from(document.querySelectorAll('button')).filter(btn => 
        btn.textContent.includes('全部') || btn.textContent.includes('全局') || btn.textContent.includes('个人')
      );
      
      const badges = Array.from(document.querySelectorAll('[class*="badge"]'));
      const globalBadges = badges.filter(b => b.textContent.includes('全局'));
      const personalBadges = badges.filter(b => b.textContent.includes('个人'));
      
      const cards = document.querySelectorAll('[class*="card"]').length;
      
      return {
        hasFilterTabs: filterTabs.length > 0,
        filterTabTexts: filterTabs.map(t => t.textContent.trim()),
        globalBadgeCount: globalBadges.length,
        personalBadgeCount: personalBadges.length,
        totalCards: cards
      };
    });
    
    console.log('页面内容分析:');
    console.log(`- 筛选标签: ${pageContent.hasFilterTabs ? '✓ 存在' : '✗ 不存在'}`);
    console.log(`  标签文本: ${pageContent.filterTabTexts.join(', ')}`);
    console.log(`- 全局徽章数量: ${pageContent.globalBadgeCount}`);
    console.log(`- 个人徽章数量: ${pageContent.personalBadgeCount}`);
    console.log(`- 卡片总数: ${pageContent.totalCards}`);
    
    // 点击"个人"筛选标签
    console.log('\n点击"个人"筛选标签...');
    const personalTabClicked = await page.evaluate(() => {
      const tabs = Array.from(document.querySelectorAll('button')).filter(btn => 
        btn.textContent.includes('个人')
      );
      if (tabs.length > 0) {
        tabs[0].click();
        return true;
      }
      return false;
    });
    
    if (personalTabClicked) {
      console.log('✓ 已点击"个人"标签');
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // 截图2：个人世界书
      const screenshot2Path = path.join(screenshotsDir, 'worldbooks-personal.png');
      await page.screenshot({ path: screenshot2Path, fullPage: true });
      console.log(`✓ 截图已保存: ${screenshot2Path}`);
      
      // 检查筛选后的内容
      const filteredContent = await page.evaluate(() => {
        const visibleCards = Array.from(document.querySelectorAll('[class*="card"]')).filter(card => {
          const style = window.getComputedStyle(card);
          return style.display !== 'none' && style.visibility !== 'hidden' && style.opacity !== '0';
        });
        
        const badges = Array.from(document.querySelectorAll('[class*="badge"]'));
        const personalBadges = badges.filter(b => b.textContent.includes('个人'));
        
        return {
          visibleCardCount: visibleCards.length,
          personalBadgeCount: personalBadges.length
        };
      });
      
      console.log('\n筛选后内容:');
      console.log(`- 可见卡片数: ${filteredContent.visibleCardCount}`);
      console.log(`- 个人徽章数: ${filteredContent.personalBadgeCount}`);
    } else {
      console.log('✗ 未找到"个人"标签');
    }

    console.log('\n=== 2. 聊天页面 - 世界书按钮 ===');
    console.log('URL: http://localhost:3000/chat/s1\n');
    
    await page.goto('http://localhost:3000/chat/s1', {
      waitUntil: 'networkidle0',
      timeout: 30000
    });
    
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // 查找世界书按钮
    const worldbookButton = await page.evaluate(() => {
      const buttons = Array.from(document.querySelectorAll('button'));
      const worldbookBtn = buttons.find(btn => 
        btn.textContent.includes('世界书') || 
        btn.querySelector('[class*="book"]') ||
        btn.querySelector('svg')
      );
      
      if (worldbookBtn) {
        return {
          found: true,
          text: worldbookBtn.textContent.trim(),
          hasIcon: worldbookBtn.querySelector('svg') !== null
        };
      }
      
      return { found: false };
    });
    
    console.log('世界书按钮:');
    if (worldbookButton.found) {
      console.log(`✓ 找到按钮`);
      console.log(`  文本: "${worldbookButton.text}"`);
      console.log(`  图标: ${worldbookButton.hasIcon ? '✓ 有' : '✗ 无'}`);
      
      // 点击世界书按钮
      console.log('\n点击世界书按钮...');
      const dialogOpened = await page.evaluate(() => {
        const buttons = Array.from(document.querySelectorAll('button'));
        const worldbookBtn = buttons.find(btn => 
          btn.textContent.includes('世界书') || 
          btn.querySelector('[class*="book"]') ||
          btn.querySelector('svg')
        );
        
        if (worldbookBtn) {
          worldbookBtn.click();
          return true;
        }
        return false;
      });
      
      if (dialogOpened) {
        console.log('✓ 已点击按钮');
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // 等待对话框出现
        try {
          await page.waitForSelector('[role="dialog"], [class*="dialog"], [class*="modal"]', { timeout: 5000 });
          console.log('✓ 对话框元素已加载');
        } catch (e) {
          console.log('⚠ 等待对话框超时，继续截图');
        }
        
        // 截图3：世界书对话框
        const screenshot3Path = path.join(screenshotsDir, 'worldbook-dialog.png');
        await page.screenshot({ path: screenshot3Path, fullPage: true });
        console.log(`✓ 截图已保存: ${screenshot3Path}`);
        
        // 检查对话框内容
        const dialogContent = await page.evaluate(() => {
          const dialog = document.querySelector('[role="dialog"]') || 
                        document.querySelector('[class*="dialog"]') ||
                        document.querySelector('[class*="modal"]') ||
                        document.querySelector('[class*="sheet"]') ||
                        document.querySelector('[class*="drawer"]');
          
          if (dialog) {
            const toggles = dialog.querySelectorAll('[role="switch"]').length;
            const worldbookItems = Array.from(dialog.querySelectorAll('[class*="item"]')).length;
            const personalBadges = Array.from(dialog.querySelectorAll('[class*="badge"]')).filter(b => 
              b.textContent.includes('个人')
            ).length;
            
            // 获取对话框的位置和大小
            const rect = dialog.getBoundingClientRect();
            
            return {
              found: true,
              toggleCount: toggles,
              itemCount: worldbookItems,
              personalBadgeCount: personalBadges,
              position: {
                x: rect.x,
                y: rect.y,
                width: rect.width,
                height: rect.height
              },
              html: dialog.outerHTML.substring(0, 500)
            };
          }
          
          return { found: false };
        });
        
        console.log('\n对话框内容:');
        if (dialogContent.found) {
          console.log(`✓ 对话框已打开`);
          console.log(`  位置: x=${dialogContent.position.x}, y=${dialogContent.position.y}`);
          console.log(`  大小: ${dialogContent.position.width}x${dialogContent.position.height}`);
          console.log(`  开关数量: ${dialogContent.toggleCount}`);
          console.log(`  世界书项数: ${dialogContent.itemCount}`);
          console.log(`  个人徽章数: ${dialogContent.personalBadgeCount}`);
          console.log(`  HTML片段: ${dialogContent.html.substring(0, 200)}...`);
        } else {
          console.log('✗ 未检测到对话框');
        }
      } else {
        console.log('✗ 点击失败');
      }
    } else {
      console.log('✗ 未找到世界书按钮');
    }

    console.log('\n=== 截图完成 ===');
    console.log(`所有截图已保存到: ${screenshotsDir}`);

  } catch (error) {
    console.error('错误:', error.message);
    throw error;
  } finally {
    await browser.close();
  }
}

takeScreenshots().catch(console.error);
