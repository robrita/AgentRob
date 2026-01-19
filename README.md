# AgentRob (Vite)

## Local development

```bash
npm install
npm run dev
```

Open http://localhost:5173

## Production build

```bash
npm run build
```

The production-ready output is in the `dist/` folder.

## GitHub Pages deployment

### User/Org Pages
- Keep `base: "./"` in [vite.config.js](vite.config.js).
- Deploy `dist/` to the `gh-pages` branch.
- URL: `https://username.github.io/`

### Project Pages
- Set `base: "/<repo-name>/"` in [vite.config.js](vite.config.js).
- Deploy `dist/` to the `gh-pages` branch.
- URL: `https://username.github.io/<repo-name>/`

## Update data
Edit [src/data/data.json](src/data/data.json) to add or update transcript messages.

## Customize styling
Update CSS variables and component styles in [src/styles.css](src/styles.css).

## Customize behavior
- UI layout: [src/components/App.js](src/components/App.js)
- Transcript rendering: [src/components/Grid.js](src/components/Grid.js)
- Message card template: [src/components/Card.js](src/components/Card.js)
- Filter/search logic: [src/components/Filter.js](src/components/Filter.js)
