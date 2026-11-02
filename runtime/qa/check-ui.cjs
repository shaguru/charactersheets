const assert = require('node:assert/strict');
const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
    const errors = [];
    page.on('pageerror', e => errors.push(String(e)));
    page.on('response', r => {
      if (r.status() >= 400 && !r.url().endsWith('/favicon.ico')) errors.push(`${r.status()} ${r.url()}`);
    });
    const jobs = [0, 1].map(i => ({
      id: `qa-${i}`, status: 'done', model: 'qwen-rapid-aio', model_name: 'Qwen Rapid AIO v19',
      mode: 'sheet', sheet: 'turnaround5', seed: 100 + i, steps: 4, cfg: 1,
      width: 768, height: 512, created: 1700000000, prompt: 'Synthetic browser test image',
      images: [{ filename: `qa-${i}.svg`, subfolder: 'QA', type: 'output' }],
    }));
    const runtime = { state: 'unloaded', queue: 0, busy: false };
    await page.route('**/forge/api/jobs?*', route => route.fulfill({ json: {
      jobs, queue_running: 0, queue_pending: 0, current: null, runtime,
    } }));
    await page.route('**/view?*', route => route.fulfill({
      contentType: 'image/svg+xml',
      body: '<svg xmlns="http://www.w3.org/2000/svg" width="768" height="512"><rect width="768" height="512" fill="#334455"/><text x="80" y="240" fill="white" font-size="40">Synthetic test image</text></svg>',
    }));
    await page.goto(process.env.FORGE_QA_URL || 'http://127.0.0.1:8288/forge/');
    await page.locator('.tile img').first().waitFor();
    assert.equal(await page.locator('#modelList input').count(), 2);
    assert.equal(await page.locator('.tile img').count(), 2);
    assert.match(await page.locator('#hw').textContent(), /^CUDA enabled/);

    const open = async () => {
      await page.locator('.tile img').first().click();
      assert.equal(await page.locator('#lightbox').isVisible(), true);
    };
    await open();
    await page.locator('#lbNext').click();
    assert.equal(await page.locator('#lightbox').isVisible(), true, 'next must keep the viewer open');
    assert.match(await page.locator('#lbMeta').textContent(), /seed 101/);
    await page.locator('#lbPrev').click();
    assert.match(await page.locator('#lbMeta').textContent(), /seed 100/);
    await page.locator('#lbTitle').click();
    assert.equal(await page.locator('#lightbox').isVisible(), true, 'sidebar must keep the viewer open');
    await page.locator('#lbImg').click();
    assert.equal(await page.locator('#lightbox').isVisible(), false, 'clicking the image must close');
    await open();
    await page.locator('#lightbox').click({ position: { x: 65, y: 3 } });
    assert.equal(await page.locator('#lightbox').isVisible(), false, 'background click must close');
    await open();
    await page.keyboard.press('Escape');
    assert.equal(await page.locator('#lightbox').isVisible(), false);
    await open();
    await page.locator('#lbClose').click();
    assert.equal(await page.locator('#lightbox').isVisible(), false);

    await page.getByRole('button', { name: /^Models/ }).click();
    assert.equal(await page.locator('.mcard').count(), 2);
    await page.getByRole('button', { name: 'Studio', exact: true }).click();
    for (const width of [1600, 1280]) {
      await page.setViewportSize({ width, height: 900 });
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false, `overflow at ${width}px`);
    }
    // Exercise the automatic 8 GB UI without requiring that physical GPU.
    await page.route('**/forge/api/config', async route => {
      const response = await route.fetch();
      const config = await response.json();
      config.memory_profile = { low_vram: true, vram_gb: 8, max_pixels: 393216 };
      for (const size of Object.values(config.presets.sizes)) {
        if (size.w && size.h) { size.w = 768; size.h = 512; }
      }
      await route.fulfill({ json: config });
    });
    await page.reload();
    await page.locator('#memoryHint').waitFor({ state: 'visible' });
    assert.match(await page.locator('#hw').textContent(), /CPU offload/);
    assert.match(await page.locator('#size option').first().textContent(), /768×512/);
    assert.deepEqual(errors, []);
    console.log('PASS: model cards, CUDA status, gallery, click-to-close, navigation, layouts, low-VRAM UI and browser errors.');
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exitCode = 1; });
