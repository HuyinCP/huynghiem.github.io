# Personal blog

Project articles are written in Markdown
and compiled to HTML with a small Node.js build script. **`index.html` is the starting page.**

## Preview on your computer

Install the build dependency once with `npm install`. After editing a project article, run
`npm run build`, then open `index.html` in a browser. For automatic refresh while editing,
install the **Live Server** extension in VS Code.

## Edit the site

- Homepage and introduction: `index.html`
- Profile photo: replace `assets/images/avt.jpg` or update its path in `index.html`.
- Project index: `projects.html` (summaries and links only)
- Project images: store them in a project folder such as `assets/images/rnn/` and update the thumbnail path in `projects.html`.
- Blog list: `blogs.html`
- Achievements page: `achievements.html` (replace the empty state with verified milestones, dates, and optional images or links)
- Knowledge Graph: `knowledge_graph.html`
- Editable project articles: one Markdown file per project inside `src/projects/`
- Shared project layout and MathJax configuration: `src/project-layout.html`
- Generated standalone project pages: `blog/` (do not edit generated project HTML directly)
- Sample article: `blog/welcome.html`
- Colors and layout: `assets/css/style.css`

To add a blog article, copy `blog/welcome.html`, edit the new file, and add a link in `blogs.html`.

To add a project:

1. Copy `src/projects/_template.md` and give it a descriptive lowercase hyphenated name.
2. Update the front matter and write the complete article, including images and MathJax equations, in that Markdown file.
3. Run `npm run build` to generate the matching standalone HTML file in `blog/`.
4. Copy a project card in `projects.html` and update its title, date, short description, thumbnail, and generated page link.
5. Add the project node, topic links, and directory entry in `knowledge_graph.html`. Keep the node URL and directory link pointed at the generated project page.

## Publish with GitHub Pages

The repository is `HuyinCP/huynghiem.github.io`, so its GitHub Pages address is **https://huyincp.github.io/huynghiem.github.io/**.

The root `.nojekyll` file tells GitHub Pages to publish the generated HTML directly. Keep it in the repository, because project Markdown files in `src/projects/` are build sources and are not meant to be rendered by Jekyll.

1. Push these files to the `main` branch.
2. Open **Settings**, then **Pages**, in the GitHub repository.
3. Select **Deploy from a branch**, branch **main**, folder **/(root)**, then click **Save**.

If you rename the repository to `HuyinCP.github.io`, the site address becomes `https://huyincp.github.io/`. The relative links in this site work with either address.


This is a static HTML, CSS, and JavaScript website inspired by
[codingowen.github.io](https://codingowen.github.io/).
