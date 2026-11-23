import { readFile, readdir, mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { marked } from "marked";

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const rootDirectory = path.resolve(scriptDirectory, "..");
const projectsDirectory = path.join(rootDirectory, "src", "projects");
const layoutPath = path.join(rootDirectory, "src", "project-layout.html");

const escapeHtml = (value = "") =>
  String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");

const PYTHON_KEYWORDS = new Set([
  "and", "as", "assert", "async", "await", "break", "class", "continue",
  "def", "del", "elif", "else", "except", "finally", "for", "from", "global",
  "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass",
  "raise", "return", "try", "while", "with", "yield"
]);

const PYTHON_CONSTANTS = new Set(["True", "False", "None"]);

const PYTHON_BUILTINS = new Set([
  "abs", "all", "any", "bool", "dict", "enumerate", "float", "int",
  "isinstance", "len", "list", "map", "max", "min", "object", "print",
  "range", "round", "set", "sorted", "str", "sum", "super", "tuple",
  "type", "zip"
]);

const PYTHON_TOKEN_PATTERN =
  /([rRbBuUfF]{0,2}(?:"""[\s\S]*?"""|'''[\s\S]*?'''))|([rRbBuUfF]{0,2}(?:"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'))|(#[^\n]*)|(\b(?:class|def)\b)(\s+)([A-Za-z_]\w*)|(\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b)|(\b[A-Za-z_]\w*\b)|(==|!=|<=|>=|\*\*|\/\/|\+=|-=|\*=|\/=|->|[=+\-*/@%<>])/g;

function wrapToken(className, text) {
  return `<span class="${className}">${escapeHtml(text)}</span>`;
}

function highlightPython(code) {
  let result = "";
  let lastIndex = 0;

  for (const match of code.matchAll(PYTHON_TOKEN_PATTERN)) {
    const startIndex = match.index;
    const fullMatch = match[0];

    if (startIndex > lastIndex) {
      result += escapeHtml(code.slice(lastIndex, startIndex));
    }

    const [
      ,
      docstring,
      stringLiteral,
      comment,
      declKeyword,
      declSpace,
      declName,
      numberLiteral,
      identifier,
      operator
    ] = match;

    if (docstring) {
      result += wrapToken("tok-docstring", docstring);
    } else if (stringLiteral) {
      result += wrapToken("tok-string", stringLiteral);
    } else if (comment) {
      result += wrapToken("tok-comment", comment);
    } else if (declKeyword) {
      const nameClass = declKeyword === "class" ? "tok-class" : "tok-function";
      result +=
        wrapToken("tok-keyword", declKeyword) +
        escapeHtml(declSpace) +
        wrapToken(nameClass, declName);
    } else if (numberLiteral) {
      result += wrapToken("tok-number", numberLiteral);
    } else if (identifier) {
      const nextChar = code[startIndex + fullMatch.length];
      if (PYTHON_KEYWORDS.has(identifier)) {
        result += wrapToken("tok-keyword", identifier);
      } else if (PYTHON_CONSTANTS.has(identifier)) {
        result += wrapToken("tok-constant", identifier);
      } else if (identifier === "self" || identifier === "cls") {
        result += wrapToken("tok-self", identifier);
      } else if (PYTHON_BUILTINS.has(identifier)) {
        result += wrapToken("tok-builtin", identifier);
      } else if (/^[A-Z][A-Za-z0-9_]*[a-z][A-Za-z0-9_]*$/.test(identifier)) {
        result += wrapToken("tok-class", identifier);
      } else if (nextChar === "(") {
        result += wrapToken("tok-call", identifier);
      } else {
        result += escapeHtml(identifier);
      }
    } else if (operator) {
      const lineStart = code.lastIndexOf("\n", startIndex - 1) + 1;
      const beforeOnLine = code.slice(lineStart, startIndex);
      if (operator === "@" && /^\s*$/.test(beforeOnLine)) {
        result += wrapToken("tok-decorator", operator);
      } else {
        result += wrapToken("tok-operator", operator);
      }
    } else {
      result += escapeHtml(fullMatch);
    }

    lastIndex = startIndex + fullMatch.length;
  }

  if (lastIndex < code.length) {
    result += escapeHtml(code.slice(lastIndex));
  }

  return result;
}

function highlightCode(code, language) {
  if (language === "python" || language === "py") {
    return highlightPython(code);
  }
  return escapeHtml(code);
}

marked.use({
  renderer: {
    code({ text, lang }) {
      const language = (lang || "").trim().toLowerCase();
      const className = language ? ` class="language-${escapeHtml(language)}"` : "";
      const highlighted = highlightCode(text, language);
      return `<pre><code${className}>${highlighted}\n</code></pre>\n`;
    }
  }
});

function parseFrontMatter(source, fileName) {
  const match = source.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n/);

  if (!match) {
    throw new Error(`${fileName} is missing front matter.`);
  }

  const metadata = {};

  for (const line of match[1].split(/\r?\n/)) {
    const separator = line.indexOf(":");

    if (separator === -1) {
      continue;
    }

    const key = line.slice(0, separator).trim();
    const value = line.slice(separator + 1).trim();
    metadata[key] = value;
  }

  return { metadata, body: source.slice(match[0].length) };
}

function formatArticleContent(content) {
  const blockTags = "blockquote|div|figure|h[1-6]|nav|ol|p|pre|table|ul";
  const adjacentBlocks = new RegExp(
    `</(${blockTags})><(?=(?:${blockTags})\\b)`,
    "g"
  );
  const lines = content.trim().replace(adjacentBlocks, "</$1>\n\n<").split(/\r?\n/);
  let insidePreformattedBlock = false;

  return lines
    .map((line) => {
      if (insidePreformattedBlock) {
        if (line.includes("</pre>")) {
          insidePreformattedBlock = false;
        }

        return line;
      }

      if (line.includes("<pre")) {
        insidePreformattedBlock = true;
      }

      const readableLine = line.startsWith("<li") ? `  ${line}` : line;
      return readableLine ? `      ${readableLine}` : "";
    })
    .join("\n");
}

function renderDifficultyTag(rawDifficulty) {
  const level = Number.parseInt(String(rawDifficulty || ""), 10);
  if (!Number.isInteger(level) || level < 1 || level > 5) {
    return "";
  }

  const filled = "★".repeat(level);
  const empty = "☆".repeat(5 - level);
  const emptyMarkup = empty
    ? `<span class="difficulty-stars-empty">${empty}</span>`
    : "";

  return `\n        <div class="article-tags">\n          <span class="difficulty-tag" aria-label="Difficulty: ${level} out of 5 stars">\n            <span class="difficulty-label">Difficulty</span>\n            <span class="difficulty-stars" aria-hidden="true">${filled}${emptyMarkup}</span>\n          </span>\n        </div>`;
}

function applyLayout(layout, metadata, content) {
  const values = {
    title: escapeHtml(metadata.title),
    description: escapeHtml(metadata.description),
    lead: escapeHtml(metadata.lead),
    date: escapeHtml(metadata.date),
    difficulty_tag: renderDifficultyTag(metadata.difficulty),
    back_url: escapeHtml(metadata.back_url || "../projects.html"),
    back_label: escapeHtml(metadata.back_label || "Projects"),
    content: formatArticleContent(content)
  };

  return layout.replace(/\{\{\s*([a-z_]+)\s*\}\}/g, (_, key) => values[key] ?? "");
}

async function buildProject(sourcePath) {
  const source = await readFile(sourcePath, "utf8");
  const { metadata, body } = parseFrontMatter(source, path.basename(sourcePath));

  for (const field of ["title", "description", "lead", "date", "output"]) {
    if (!metadata[field]) {
      throw new Error(`${path.basename(sourcePath)} is missing '${field}'.`);
    }
  }

  const content = await marked.parse(body, { gfm: true });
  const layout = await readFile(layoutPath, "utf8");
  const sourceReference = path.relative(rootDirectory, sourcePath).replaceAll("\\", "/");
  const output = `<!-- Generated by npm run build. Edit ${sourceReference} instead. -->\n${applyLayout(layout, metadata, content)}`;
  const outputPath = path.resolve(rootDirectory, metadata.output);

  if (!outputPath.startsWith(`${rootDirectory}${path.sep}`)) {
    throw new Error(`Invalid output path: ${metadata.output}`);
  }

  await mkdir(path.dirname(outputPath), { recursive: true });
  await writeFile(outputPath, output, "utf8");
  return path.relative(rootDirectory, outputPath).replaceAll("\\", "/");
}

const projectFiles = (await readdir(projectsDirectory))
  .filter((fileName) => fileName.endsWith(".md") && !fileName.startsWith("_"))
  .sort();

if (!projectFiles.length) {
  throw new Error("No project source files were found.");
}

for (const projectFile of projectFiles) {
  const outputPath = await buildProject(path.join(projectsDirectory, projectFile));
  console.log(`Built ${outputPath}`);
}
