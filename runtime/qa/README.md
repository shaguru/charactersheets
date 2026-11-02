Optional UI regression check (requires Node.js, Microsoft Edge and a running server):

1. Start Character Forge.
2. In `runtime/qa`, run `npm install`, then `npm test`.

Checks model count, gallery, CUDA status, image/background click-to-close, navigation, sidebar behavior, Escape, the close button, JavaScript/network errors and horizontal overflow at 1600px and 1280px. Synthetic images and gallery entries are injected through browser request interception; no image files or projects are created or modified. Set `FORGE_QA_URL` to test another port.
