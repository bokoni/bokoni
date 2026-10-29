// The Bokoni language for Monaco: highlighting (Monarch), brackets, indentation,
// completion of keywords and snippets.

import type * as Monaco from "monaco-editor/editor";

export const LANGUAGE = "bokoni";

// The lexer's reserved words (`src/lexer.rs`).
const keywords = [
  "let", "def", "end", "if", "elsif", "else", "unless", "then", "case", "when", "while",
  "for", "in", "do", "break", "next", "return", "begin", "rescue", "ensure", "raise",
  "super", "trait", "abstract", "include", "forall", "private", "protected", "type",
  "class", "enum", "annotation", "macro", "module", "require", "and", "or", "not",
];
const constants = ["true", "false", "nil", "self"];
const builtinTypes = ["int", "float", "str", "bool", "any"];

const tokens: Monaco.languages.IMonarchLanguage = {
  defaultToken: "",
  keywords,
  constants,
  builtinTypes,
  tokenizer: {
    root: [
      [/#.*$/, "comment"],
      [/@\[/, { token: "annotation", next: "@annotation" }],
      [/@[a-z_]\w*/, "variable.instance"],
      [/(def)(\s+)([a-zA-Z_]\w*[?!]?)/, ["keyword", "white", "function"]],
      [/[A-Z][A-Z0-9_]+\b/, "constant"],
      [/[A-Z]\w*/, "type"],
      [
        /[a-z_]\w*[?!]?/,
        {
          cases: {
            "@keywords": "keyword",
            "@constants": "constant",
            "@builtinTypes": "type",
            "@default": "identifier",
          },
        },
      ],
      [/\d[\d_]*\.\d[\d_]*([eE][+-]?\d+)?/, "number.float"],
      [/\d[\d_]*/, "number"],
      [/"/, { token: "string.quote", next: "@string" }],
      [/[{}()[\]]/, "@brackets"],
      [/\|/, "delimiter"],
      [/[=<>!+\-*/%?:.&^~]+/, "operator"],
      [/[,;]/, "delimiter"],
    ],
    string: [
      [/[^\\"#]+/, "string"],
      [/\\./, "string.escape"],
      [/#\{/, { token: "delimiter.interpolation", next: "@interpolation" }],
      [/#/, "string"],
      [/"/, { token: "string.quote", next: "@pop" }],
    ],
    interpolation: [
      [/\{/, { token: "@brackets", next: "@braces" }],
      [/\}/, { token: "delimiter.interpolation", next: "@pop" }],
      { include: "root" },
    ],
    braces: [
      [/\{/, { token: "@brackets", next: "@braces" }],
      [/\}/, { token: "@brackets", next: "@pop" }],
      { include: "root" },
    ],
    annotation: [
      [/\]/, { token: "annotation", next: "@pop" }],
      [/[A-Z]\w*/, "annotation"],
      { include: "root" },
    ],
  },
};

// A line opening a block (`… end`): not a one-line form (`def f() = x`, `if c then x end`).
const opens = new RegExp(
  [
    String.raw`^\s*(class|type|enum|trait|module|annotation|macro|while|for|begin)\b(?!.*\bend\s*$)`,
    String.raw`^\s*(abstract\s+|private\s+|protected\s+)*def\b(?!.*\)\s*(:[^=]*)?=\s*\S)(?!.*\bend\s*$)`,
    String.raw`^\s*(if|unless|elsif|else|case|rescue|ensure)\b(?!.*\bthen\s+\S)(?!.*\bend\s*$)`,
    String.raw`^\s*when\b(?!.*\bthen\s+\S)`,
    String.raw`=\s*(if|case|begin)\b(?!.*\bend\s*$)`,
    String.raw`\bdo(\s*\|[^|]*\|)?\s*$`,
  ].join("|"),
);

const config: Monaco.languages.LanguageConfiguration = {
  comments: { lineComment: "#" },
  brackets: [
    ["{", "}"],
    ["[", "]"],
    ["(", ")"],
  ],
  autoClosingPairs: [
    { open: "{", close: "}" },
    { open: "[", close: "]" },
    { open: "(", close: ")" },
    { open: '"', close: '"', notIn: ["string", "comment"] },
  ],
  surroundingPairs: [
    { open: "{", close: "}" },
    { open: "[", close: "]" },
    { open: "(", close: ")" },
    { open: '"', close: '"' },
  ],
  wordPattern: /[a-zA-Z_]\w*[?!]?/,
  indentationRules: {
    increaseIndentPattern: opens,
    decreaseIndentPattern: /^\s*(end|else|elsif|when|rescue|ensure)\b/,
  },
};

const snippets: [label: string, body: string, doc: string][] = [
  ["def", "def ${1:name}(${2:x}: ${3:int}): ${4:int}\n\t$0\nend", "function"],
  ["class", "class ${1:Name}\n\tdef init(@${2:x}: ${3:int})\n\tend\n\t$0\nend", "class"],
  ["type", "type ${1:Name}\n\t${2:x}: ${3:int} = ${4:0}\nend", "data type"],
  ["enum", "enum ${1:Name}\n\t${2:A}\n\t${3:B}\nend", "enum"],
  ["if", "if ${1:cond}\n\t$0\nend", "if … end"],
  ["for", "for ${1:x} in ${2:xs}\n\t$0\nend", "for … in"],
  ["case", "case ${1:x}\nwhen ${2:1} then ${3:a}\nelse ${4:b}\nend", "case … when"],
  ["begin", "begin\n\t$0\nrescue e: Error\n\te.message\nend", "begin … rescue"],
];

export function registerBokoni(monaco: typeof Monaco) {
  monaco.languages.register({ id: LANGUAGE, extensions: [".bk"], aliases: ["Bokoni"] });
  monaco.languages.setMonarchTokensProvider(LANGUAGE, tokens);
  monaco.languages.setLanguageConfiguration(LANGUAGE, config);
  monaco.languages.registerCompletionItemProvider(LANGUAGE, {
    provideCompletionItems(model, position) {
      const word = model.getWordUntilPosition(position);
      const range = {
        startLineNumber: position.lineNumber,
        endLineNumber: position.lineNumber,
        startColumn: word.startColumn,
        endColumn: word.endColumn,
      };
      const kinds = monaco.languages.CompletionItemKind;
      const rule = monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet;
      return {
        suggestions: [
          ...snippets.map(([label, body, doc]) => ({
            label,
            kind: kinds.Snippet,
            insertText: body,
            insertTextRules: rule,
            detail: doc,
            range,
          })),
          ...[...keywords, ...constants, ...builtinTypes].map((k) => ({
            label: k,
            kind: kinds.Keyword,
            insertText: k,
            range,
          })),
        ],
      };
    },
  });
}

/** Colours of the tokens above, for a light and a dark theme. */
export function defineThemes(monaco: typeof Monaco) {
  monaco.editor.defineTheme("bokoni-light", {
    base: "vs",
    inherit: true,
    rules: [
      { token: "keyword", foreground: "8b2fc9", fontStyle: "bold" },
      { token: "type", foreground: "0f7b6c" },
      { token: "constant", foreground: "b35900" },
      { token: "function", foreground: "1a5fb4" },
      { token: "variable.instance", foreground: "a51d2d" },
      { token: "annotation", foreground: "986801" },
      { token: "string", foreground: "2b7a0b" },
      { token: "string.escape", foreground: "b35900" },
      { token: "delimiter.interpolation", foreground: "8b2fc9" },
      { token: "number", foreground: "b35900" },
      { token: "comment", foreground: "7a7f87", fontStyle: "italic" },
      { token: "operator", foreground: "5c6370" },
    ],
    colors: { "editor.background": "#fbfaf8" },
  });
  monaco.editor.defineTheme("bokoni-dark", {
    base: "vs-dark",
    inherit: true,
    rules: [
      { token: "keyword", foreground: "c792ea", fontStyle: "bold" },
      { token: "type", foreground: "7fdbca" },
      { token: "constant", foreground: "f78c6c" },
      { token: "function", foreground: "82aaff" },
      { token: "variable.instance", foreground: "ff7a93" },
      { token: "annotation", foreground: "ffcb6b" },
      { token: "string", foreground: "c3e88d" },
      { token: "string.escape", foreground: "f78c6c" },
      { token: "delimiter.interpolation", foreground: "c792ea" },
      { token: "number", foreground: "f78c6c" },
      { token: "comment", foreground: "697098", fontStyle: "italic" },
      { token: "operator", foreground: "89ddff" },
    ],
    colors: { "editor.background": "#16171d" },
  });
}
