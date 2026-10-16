# Project Rules

These rules apply to the entire repository.

## Language

- All user-facing website content must be written in English.
- This includes page titles, headings, navigation labels, buttons, descriptions, dates, image captions, image alternative text, placeholders, helper messages, and article content.
- Use English for HTML comments, code comments, file names, CSS class names, JavaScript identifiers, and documentation added to the repository.
- Do not add Vietnamese text to the website unless the user explicitly requests bilingual or Vietnamese content for a specific task.
- The assistant may continue speaking Vietnamese with the user; this language rule applies to project files and the website interface.

## Consistency

- When editing a page, check any new or modified user-facing text on that page and keep it consistently in English.
- Prefer clear, natural English over literal translation.
- Preserve working relative links and the existing visual style unless the task requires a design change.

## Content Structure

- Keep `projects.html` as a project index only. It may contain a title, date, short description, thumbnail, and link for each project, but it must not contain the full project article.
- Store the editable source for every project in its own Markdown file inside `src/projects/`.
- Use one source file per project, with a descriptive lowercase hyphenated file name such as `src/projects/rnn-from-scratch.md`.
- Store reusable equation fragments in `src/includes/equations/<project-name>/` and include them from the Markdown source.
- Run `npm run build` after editing project source or equation files. The build writes the final standalone HTML files into `blog/`.
- Treat generated project HTML files inside `blog/` as build output. Do not edit them directly.
- Keep `blogs.html` as a technical blog index only. Each listed blog post must link to its own standalone HTML file inside `blog/`.
- Never combine multiple full project articles or blog posts into `projects.html`, `blogs.html`, or a single shared source file.
