# Personal blog

A personal website built with Jekyll and hosted on GitHub Pages.

## Publish the website

The current repository is `HuyinCP/GitHubIO`, so the default address is **https://huyincp.github.io/GitHubIO/**.

1. Push these files to the `main` branch.
2. In the repository, open **Settings → Pages**.
3. Under **Build and deployment**, select **Source: Deploy from a branch**.
4. Select **Branch: main** and **/(root)**, then click **Save**.
5. Wait a few minutes and visit the address above.

If you rename the repository to `HuyinCP.github.io`, change the setting to `baseurl: ""` in `_config.yml`. The site address will then be `https://huyincp.github.io/`.

## Edit the content

- Site title and description: `_config.yml`.
- Introduction: `index.md`.
- Projects: `projects.md`.
- Knowledge Graph: `knowledge_graph.md`. It displays topics from the `tags` in your blog posts.
- Blog posts: create a file in `_posts` named `YYYY-MM-DD-post-title.md`; use the sample post as a guide. Add `tags: [Topic Name]` to include it in the Knowledge Graph.
- Colors and layout: `assets/css/style.css`.

For a local preview (optional), install Ruby and Bundler, then run `bundle install` and `bundle exec jekyll serve`. You can also edit files directly on GitHub; Pages will rebuild the site after each update.
