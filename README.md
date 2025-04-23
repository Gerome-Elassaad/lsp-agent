<div align="center">
   <picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://www.qodo.ai/wp-content/uploads/2024/09/15-Best-AI-Coding-Assistant-Tools-in-2025-1.png">
  <img alt="Logo" src="" width="650em">
   </picture>
</div>

<p align="center">
   <p align="center"><b>Empowering not replacing programmers for procise & efficiant workflows.</b></p>
</p>

<p align="center">
| <a href="https://github.com/Gerome-Elassaad/lsp-code-client/wiki"><b>Documentation</b></a> 
</p>

---

lsp-code-client is an open source [language server](https://microsoft.github.io/language-server-protocol/) that serves as a backend for AI-powered functionality in your favorite code editors. It offers features like in-editor chatting with LLMs and code completions. Because it is a language server, it works with any editor that has LSP support.

**The goal of lsp-code-client is to assist and empower software engineers by integrating with the tools they already know and love, not replace software engineers.**

A short list of a few of the editors it works with:
- VS Code
- NeoVim
- Emacs
- Helix
- Sublime

It works with many many many more editors.

**NOTE: This project is currently used daily by many users and has reached a stage where it has all the features I want for it. Development is not necessarily done, but no new features are currently being developed for it.**

# Features

## In-Editor Chatting

Chat directly in your codebase with your favorite local or hosted models.

![in-editor-chatting](https://github.com/user-attachments/assets/c69a9dc0-c0ac-4786-b24b-f5b5d19ffd3a)

*Chatting with Claude Sonnet in Helix*

## Custom Actions

Create custom actions to do code refactoring, code completions and more!

![custom-actions](https://github.com/user-attachments/assets/6522dced-d5ee-43bc-8b64-f4313bcc82f2)

*Using Claude Sonnet to perform refactoring with chain of thought prompting in Helix*

## Code Completions

lsp-code-client can work as an alternative to Github Copilot.

https://github.com/Gerome-Elassaad/lsp-code-client/assets/19626586/59430558-da23-4991-939d-57495061c21b

*On the left: VS Code using Mistral Codestral. On the right: Helix using stabilityai/stable-code-3b*

**Note that speed for completions is entirely dependent on the backend being used. For the fastest completions we recommend using either a small local model or Groq.**

# Documentation

See the wiki for instructions on:
- [Getting Started](https://github.com/Gerome-Elassaad/lsp-code-client/wiki)
- [Installation](https://github.com/Gerome-Elassaad/lsp-code-client/wiki/Installation)
- [Configuration](https://github.com/Gerome-Elassaad/lsp-code-client/wiki/Configuration)
- [In-Editor Chatting](https://github.com/Gerome-Elassaad/lsp-code-client/wiki/In%E2%80%90Editor-Chatting)
- [Plugins](https://github.com/Gerome-Elassaad/lsp-code-client/wiki/Plugins)
- [Server Capabilities](https://github.com/Gerome-Elassaad/lsp-code-client/wiki/Server-Capabilities-and-Functions)
- [and more](https://github.com/Gerome-Elassaad/lsp-code-client/wiki)

# The Case for lsp-code-client

**tl;dr lsp-code-client abstracts complex implementation details from editor specific plugin authors, centralizing open-source development work into one shareable backend.**

Editor integrated AI-powered assistants are here to stay. They are not perfect, but are only improving and [early research is already showing the benefits](https://arxiv.org/pdf/2206.15331). While several companies have released advanced AI-powered editors like [Cursor](https://cursor.sh/), the open-source community lacks a direct competitor.

lsp-code-client aims to fill this gap by providing a language server that integrates AI-powered functionality into the editors we know and love. Here’s why we believe lsp-code-client is necessary and beneficial:

1. **Unified AI Features**:
    - By centralizing AI features into a single backend, lsp-code-client allows supported editors to benefit from these advancements without redundant development efforts.

2. **Simplified Plugin Development**:
    - lsp-code-client abstracts away the complexities of setting up LLM backends, building complex prompts and soon much more. Plugin developers can focus on enhancing the specific editor they are working on, rather than dealing with backend intricacies.

3. **Enhanced Collaboration**:
    - Offering a shared backend creates a collaborative platform where open-source developers can come together to add new functionalities. This unified effort fosters innovation and reduces duplicated work.

4. **Broad Compatibility**:
    - lsp-code-client supports any editor that adheres to the Language Server Protocol (LSP), ensuring that a wide range of editors can leverage the AI capabilities provided by lsp-code-client.

5. **Flexible LLM Backend Support**:
    - Currently, lsp-code-client supports llama.cpp, Ollama, OpenAI-compatible APIs, Anthropic-compatible APIs, Gemini-compatible APIs and Mistral AI FIM-compatible APIs, giving developers the flexibility to choose their preferred backend. This list will soon grow.

6. **Future-Ready**:
    - lsp-code-client is committed to staying updated with the latest advancements in LLM-driven software development.
