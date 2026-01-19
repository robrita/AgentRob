---
agent: agent
type: project-refactoring
complexity: high
duration: multi-step
---

# Convert Monolithic HTML to Modular Vite Project

## Objective
Transform a large, static HTML file (2000+ lines with inline styles, data, and scripts) into a modern, maintainable Vite-based project with modular architecture, separated concerns, and production-ready build tooling.

## Input Requirements
- Existing monolithic HTML file containing:
  - Inline `<style>` tag with all CSS
  - JavaScript data array or object with asset/catalog data
  - `<script>` tags with filtering, rendering, and event handling logic
  - Complete HTML structure with header, sidebar, gallery/list, and metadata
- Clear understanding of:
  - Data structure (properties, types, array format)
  - Filter logic (groups, OR/AND combinations, search capability)
  - Component boundaries (what are distinct UI sections?)
  - Styling approach (color system, responsive breakpoints)

## Architecture Design

### Project Structure
```
project-root/
├── package.json                 (dependencies, scripts)
├── vite.config.js              (Vite build configuration)
├── index.html                   (10-line entry point)
└── src/
    ├── main.js                  (application bootstrap)
    ├── styles.css               (all CSS extracted)
    ├── data/
    │   └── data.json            (data array as JSON)
    ├── components/
    │   ├── App.js               (root component/template)
    │   ├── Filter.js            (filter UI + logic)
    │   ├── Card.js              (card/item factory)
    │   └── Grid.js              (grid/list rendering)
    └── utils/
        ├── icons.js             (SVG icon definitions)
        ├── constants.js         (color/config mappings)
        └── formatters.js        (string utilities)
```

### Core Principles
1. **Separation of Concerns**: Data, styles, logic, components completely separated
2. **Modular Components**: Each UI element is a self-contained function returning HTML/DOM
3. **Single Data Source**: All data in one JSON file, easy to edit without touching code
4. **Vanilla JS**: No framework dependencies for simplicity and lightweight output
5. **Utility Modules**: Icons, constants, formatters in standalone modules for reusability
6. **CSS Variables**: Use CSS custom properties for theming and maintainability

## Implementation Checklist

### Phase 1: Project Initialization
- [ ] Create `package.json` with Vite 5.x, dev/build/preview scripts
- [ ] Create `vite.config.js` with base path configuration (relative './' or configurable)
- [ ] Create minimal `index.html` with single `<div id="app">` and module script

### Phase 2: Data & Styles Extraction
- [ ] Extract all CSS from `<style>` tag into `src/styles.css`
- [ ] Use CSS custom properties (--colors, --spacing, --shadows)
- [ ] Create `src/data/` directory
- [ ] Convert inline data array to `src/data/data.json` with preserved structure
- [ ] Ensure all data properties map to UI display requirements

### Phase 3: Utility Modules
- [ ] Create `src/utils/icons.js` with SVG strings for all icon types
- [ ] Create `src/utils/constants.js` with color/type mappings, configuration
- [ ] Create `src/utils/formatters.js` with string transformation helpers (title case, normalize, parse lists)

### Phase 4: Component Architecture
- [ ] **App.js**: Root component returning page layout HTML (header, sidebar, content structure)
- [ ] **Filter.js**: Two functions:
  - `Filters()` - returns filter UI HTML for sidebar
  - `setupFilters({data})` - attaches event listeners, implements filter/search logic
- [ ] **Card.js**: `createCard(item)` factory returning DOM element with all metadata
- [ ] **Grid.js**: `renderData(data, container)` populates container, returns array of elements

### Phase 5: Bootstrap & Integration
- [ ] Create `src/main.js` that:
  1. Imports styles.css globally
  2. Loads data.json
  3. Mounts App component to #app
  4. Calls renderData() with data
  5. Calls setupFilters() with data reference
- [ ] Verify all imports resolve and no errors in console

### Phase 6: Documentation
- [ ] Create comprehensive `README.md` with:
  - Local development: `npm install && npm run dev`
  - Production build: `npm run build → dist/`
  - GitHub Pages deployment (user/org pages vs project subpath)
  - Data updates guide (edit data.json)
  - Customization guide (styles.css, main.js)
- [ ] Add comments to complex functions (filter logic, component factories)

## Filter Logic Specification
Implement OR within groups (select multiple filters in same category returns union) and AND between groups (all selected filters must match):

**Example Groups:**
- **Category**: [Data, Analytics, AI App, AI Agent] → OR logic
- **Type**: [Code, Blog, Demo, Documentation, etc.] → OR logic
- **Visibility**: [Public, Internal] → OR logic
- **Search**: Text search across title/description (AND with selected filters)

## Styling Approach
- Use CSS variables for all colors: `--bg`, `--panel`, `--ink`, `--brand`, `--shadow`, `--radius`
- Responsive grid: sidebar (280px fixed) + content (1fr), stacks below 880px breakpoint
- Tag/badge color variants mapped via classes (code, design, migration, blog, demo, etc.)
- Card hover effects, visibility badges, form input styles all in single CSS file

## Success Criteria
✅ Project runs locally: `npm run dev` → accessible on localhost
✅ No build errors: `npm run build` → clean dist/ output
✅ All modules import correctly, no console errors
✅ Filter logic works: OR within groups, AND between groups, search functional
✅ Responsive design: Works on mobile and desktop
✅ Data easily editable: Change data.json, rebuild/refresh to see updates
✅ Documentation complete: Local dev, build, GitHub Pages deployment, customization all documented
✅ No framework dependencies: Pure JavaScript, CSS, HTML only

## Customization Points
Users should be able to:
- Edit `src/data/data.json` to add/remove/modify items
- Change colors in `src/styles.css` (CSS variables)
- Update filter categories/types in `src/utils/constants.js` and icons in `src/utils/icons.js`
- Modify card layout in `src/components/Card.js`
- Adjust responsive breakpoints in `src/styles.css`

## Deployment Options

### Local Development
```bash
cd project-root
npm install
npm run dev
# Open http://localhost:5173
```

### Production Build
```bash
npm run build
# dist/ folder ready for deployment
```

### GitHub Pages (User/Org Pages)
- Keep `base: './'` in vite.config.js
- Deploy `dist/` to `gh-pages` branch
- Will be available at `https://username.github.io/`

### GitHub Pages (Project Pages)
- Change `base: '/<repo-name>/'` in vite.config.js
- Deploy `dist/` to `gh-pages` branch
- Will be available at `https://username.github.io/<repo-name>/`

## Expected Outcome
A modular, production-ready web application with:
- Clean, maintainable source code
- Separated data layer (JSON)
- Component-based architecture
- Full Vite build pipeline
- Comprehensive deployment documentation
- Easy customization and data updates