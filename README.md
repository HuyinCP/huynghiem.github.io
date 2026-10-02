# Personal blog

This is a plain HTML, CSS, and JavaScript website inspired by the layout of [codingowen.github.io](https://codingowen.github.io/). **`index.html` is the starting page.** The layout was written for this site; the reference site's source code, photo, and article text were not copied.

## Preview on your computer

Open `index.html` in a browser. For automatic refresh while editing, install the **Live Server** extension in VS Code, then right-click `index.html` and choose **Open with Live Server**. If that menu item is missing, the extension is not installed or enabled.

## Edit the site

- Homepage and introduction: `index.html`
- Profile photo: replace `assets/images/profile-placeholder.svg` with your own image and update its path in `index.html`.
- Projects: `projects.html`
- Project image: replace `assets/images/project-placeholder.svg` with your own image and update its path in `projects.html`.
- Blog list: `blogs.html`
- Knowledge Graph: `knowledge_graph.html`
- Sample article: `blog/welcome.html`
- Colors and layout: `assets/css/style.css`

To add an article, copy `blog/welcome.html`, edit the new file, add a link in `blogs.html`, and add its title, URL, and topics to the `graph-data` JSON in `knowledge_graph.html`.

## Publish with GitHub Pages

The current repository is `HuyinCP/GitHubIO`, so its GitHub Pages address will be **https://huyincp.github.io/GitHubIO/**.

1. Push these files to the `main` branch.
2. Open **Settings → Pages** in the GitHub repository.
3. Select **Deploy from a branch**, branch **main**, folder **/(root)**, then click **Save**.

If you rename the repository to `HuyinCP.github.io`, the site address becomes `https://huyincp.github.io/`. The relative links in this site work with either address.
