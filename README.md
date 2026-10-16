# Personal blog

This is a static HTML, CSS, and JavaScript website inspired by the layout of. Project articles are written in Markdown and compiled to HTML with a small Node.js build script. **`index.html` is the starting page.**

https://codingowen.github.io/

## Preview on your computer

Install the build dependency once with `npm install`. After editing a project article or equation, run `npm run build`, then open `index.html` in a browser. For automatic refresh while editing, install the **Live Server** extension in VS Code.

## Edit the site

- Homepage and introduction: `index.html`
- Profile photo: replace `assets/images/profile-placeholder.svg` with your own image and update its path in `index.html`.
- Project index: `projects.html` (summaries and links only)
- Project image: replace `assets/images/project-placeholder.svg` with your own image and update its path in `projects.html`.
- Blog list: `blogs.html`
- Knowledge Graph: `knowledge_graph.html`
- Editable project articles: one Markdown file per project inside `src/projects/`
- Reusable project equations: `src/includes/equations/`
- Generated standalone project pages: `blog/` (do not edit generated project HTML directly)
- Sample article: `blog/welcome.html`
- Colors and layout: `assets/css/style.css`

To add an article, copy `blog/welcome.html`, edit the new file, add a link in `blogs.html`, and add its title, URL, and topics to the `graph-data` JSON in `knowledge_graph.html`.

To add a project, copy `src/projects/_template.md`, rename it, and update its front matter and content. Add equation fragments under `src/includes/equations/`, using `src/includes/equations/_template.html` as a starting point. Run `npm run build`, then add only the title, date, short description, thumbnail, and generated page link to `projects.html`.

## Publish with GitHub Pages

The current repository is `HuyinCP/GitHubIO`, so its GitHub Pages address will be **https://huyincp.github.io/GitHubIO/**.

1. Push these files to the `main` branch.
2. Open **Settings → Pages** in the GitHub repository.
3. Select **Deploy from a branch**, branch **main**, folder **/(root)**, then click **Save**.

If you rename the repository to `HuyinCP.github.io`, the site address becomes `https://huyincp.github.io/`. The relative links in this site work with either address.

[codingowen.github.io](https://codingowen.github.io/)