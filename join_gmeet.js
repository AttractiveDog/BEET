const puppeteer = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');
const { spawn } = require('child_process');
const path = require('path');
puppeteer.use(StealthPlugin());

// TODO: Replace these with your actual credentials and meeting link
const MEET_URL = 'https://meet.google.com/rfn-jhrv-zow'; // e.g., https://meet.google.com/abc-defg-hij'
const EMAIL = 'vext921@gmail.com';
const PASSWORD = 'Vext@Pass25625';

// Launch BEET bot GUI in a new terminal window (konsole)
spawn('konsole', ['-e', 'bash', 'run_beet_bot.sh'], {
  cwd: path.resolve(__dirname, 'BEET'),
  detached: true,
  stdio: 'ignore'
}).unref();

// Start meeting automation immediately
(async () => {
  const browser = await puppeteer.launch({
    headless: false, // Must be false to avoid detection
    args: [
      '--use-fake-ui-for-media-stream', // Auto-allow mic/cam
      '--no-sandbox',
      '--disable-setuid-sandbox'
    ],
    defaultViewport: null
  });

  const page = await browser.newPage();

  // Go to Google login
  await page.goto('https://accounts.google.com/signin', { waitUntil: 'networkidle2' });

  // Login
  await page.type('input[type="email"]', EMAIL, { delay: 50 });
  await page.click('#identifierNext');
  await page.waitForTimeout(2000);
  await page.type('input[type="password"]', PASSWORD, { delay:50 });
  await page.click('#passwordNext');
  await page.waitForNavigation({ waitUntil: 'networkidle2' });

  // Go to Google Meet
  await page.goto(MEET_URL, { waitUntil: 'networkidle2' });

  // Handle 'Try desktop notifications' popup if it appears
  const notifSelector = 'button[aria-label="Not now"], div[role="dialog"] button';
  try {
    await page.waitForSelector(notifSelector, { visible: true, timeout: 4000 });
    const notifButtonText = await page.evaluate((sel) => {
      const btns = Array.from(document.querySelectorAll(sel));
      const btn = btns.find(b => b.innerText && b.innerText.match(/not now/i));
      if (btn) btn.click();
      return btn ? btn.innerText : null;
    }, notifSelector);
    if (notifButtonText) {
      console.log('Clicked "Not now" for desktop notifications');
      await page.waitForTimeout(1000);
    }
  } catch (e) {
    // Popup did not appear, continue
  }

  // Check if a name input is present (for guest join)
  const nameInputSelector = 'input[aria-label="Your name"], input[name="identifier"]';
  const nameInput = await page.$(nameInputSelector);
  if (nameInput) {
    console.log('Name input detected, entering name...');
    await page.type(nameInputSelector, 'Bot User', { delay: 50 });
    // Find and click the join button by searching all buttons for the correct text
    const joinButtonTextVariants = [/join now/i, /ask to join/i, /join/i];
    let joined = false;
    const allButtons = await page.$$('button');
    for (const btn of allButtons) {
      const text = await page.evaluate(el => {
        // Check all text content in button and its children
        return el.innerText || Array.from(el.querySelectorAll('*')).map(e => e.innerText).join(' ');
      }, btn);
      if (joinButtonTextVariants.some(re => re.test(text))) {
        await btn.click();
        console.log(`Clicked join button with text: ${text}`);
        joined = true;
        break;
      }
    }
    if (!joined) {
      console.log('Join button not found!');
    }
  } else {
    // Wait for the "Join now" button and click it (for logged-in users)
    try {
      const joinButtonTextVariants = [/join now/i, /ask to join/i, /join/i];
      let joined = false;
      const allButtons = await page.$$('button');
      for (const btn of allButtons) {
        const text = await page.evaluate(el => {
          return el.innerText || Array.from(el.querySelectorAll('*')).map(e => e.innerText).join(' ');
        }, btn);
        if (joinButtonTextVariants.some(re => re.test(text))) {
          await btn.click();
          console.log(`Clicked join button with text: ${text}`);
          joined = true;
          break;
        }
      }
      if (!joined) {
        console.log('Join button not found for logged-in user!');
      }
    } catch (e) {
      console.log('Error while trying to click join button for logged-in user!');
    }
  }

  // Wait a bit to ensure you joined
  await page.waitForTimeout(5000);

  // You can now interact further, or close the browser
  // await browser.close();
})(); 