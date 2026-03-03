const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

async function testDialog() {
  const browser = await puppeteer.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1920, height: 1080 });

    console.log('导航到聊天页面...');
    await page.goto('http://localhost:3000/chat/s1', {
      waitUntil: 'networkidle0',
      timeout: 30000
    });
    
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // 查找所有可能的世界书相关按钮
    const buttons = await page.evaluate(() => {
      const allButtons = Array.from(document.querySelectorAll('button'));
      return allButtons.map((btn, idx) => ({
        index: idx,
        text: btn.textContent.trim(),
        hasBookIcon: btn.innerHTML.includes('book') || btn.innerHTML.includes('Book'),
        hasSvg: btn.querySelector('svg') !== null,
        className: btn.className,
        ariaLabel: btn.getAttribute('aria-label')
      })).filter(b => 
        b.text.includes('世界书') || 
        b.hasBookIcon || 
        b.ariaLabel?.includes('世界书') ||
        b.ariaLabel?.includes('worldbook')
      );
    });
    
    console.log('\n找到的世界书相关按钮:');
    console.log(JSON.stringify(buttons, null, 2));
    
    if (buttons.length > 0) {
      console.log(`\n点击第一个按钮 (索引 ${buttons[0].index})...`);
      
      await page.evaluate((idx) => {
        const allButtons = Array.from(document.querySelectorAll('button'));
        allButtons[idx].click();
      }, buttons[0].index);
      
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // 检查所有可能的对话框/抽屉/sheet
      const dialogInfo = await page.evaluate(() => {
        const selectors = [
          '[role="dialog"]',
          '[data-state="open"]',
          '[class*="sheet"]',
          '[class*="drawer"]',
          '[class*="modal"]',
          '[class*="dialog"]'
        ];
        
        const results = [];
        
        for (const selector of selectors) {
          const elements = document.querySelectorAll(selector);
          elements.forEach(el => {
            const rect = el.getBoundingClientRect();
            const isVisible = rect.width > 0 && rect.height > 0;
            
            if (isVisible) {
              results.push({
                selector,
                tagName: el.tagName,
                className: el.className,
                role: el.getAttribute('role'),
                dataState: el.getAttribute('data-state'),
                position: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
                textContent: el.textContent.substring(0, 500),
                innerHTML: el.innerHTML.substring(0, 1000)
              });
            }
          });
        }
        
        return results;
      });
      
      console.log('\n找到的对话框/抽屉:');
      console.log(JSON.stringify(dialogInfo, null, 2));
      
      // 截图
      const screenshotsDir = path.join(__dirname, 'screenshots');
      const screenshotPath = path.join(screenshotsDir, 'dialog-debug.png');
      await page.screenshot({ path: screenshotPath, fullPage: true });
      console.log(`\n截图已保存: ${screenshotPath}`);
    }

  } catch (error) {
    console.error('错误:', error);
  } finally {
    await browser.close();
  }
}

testDialog().catch(console.error);
