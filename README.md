# Open Agent Platform LangGraph Tools Agent

A pre-built LangGraph tools agent for Open Agent Platform. It contains support for MCP servers and a LangConnect RAG tool.

> [!TIP]
> This project is built for [Open Agent Platform](https://github.com/langchain-ai/open-agent-platform), a citizen developer platform for building, testing, and using agents.

## Setup

First, clone the repository and create a new virtual environment:

```bash
git clone https://github.com/langchain-ai/oap-langgraph-tools-agent.git
cd oap-langgraph-tools-agent
```

```bash
uv venv
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
uv sync
```

Then set the environment variables:

```bash
cp .env.example .env
```

This project requires a Supabase account with authentication to be setup. This is because this project implements custom LangGraph authentication so that it can be called directly from a web client.

After setting your environment variables, you can start the server by running:

```bash
# The --no-browser will disable auto-opening LangGraph studio when the server starts
# optional, but recommended since the studio is not needed for this project
uv run langgraph dev --no-browser
```

The server will now be running on `http://localhost:2024`.

## Open Agent Platform

This agent has been configured to work with the [Open Agent Platform](https://github.com/langchain-ai/open-agent-platform). Please see the [OAP docs](https://github.com/langchain-ai/open-agent-platform/tree/main/README.md) for more information on how to add this agent to your OAP instance.

To update the OAP configuration, you can modify the `GraphConfigPydantic` class in the `agent.py` file. OAP will automatically register any changes to this class. You can modify a specific field's properties by editing the `x_oap_ui_config` metadata object. For more information, see the [Open Agent Platform documentation on graph configuration](https://github.com/langchain-ai/open-agent-platform/?tab=readme-ov-file#configuration).

## Authentication

This project uses LangGraph custom auth to authenticate requests to the server. It's configured to use Supabase as the authentication provider, however it can be easily swapped for another service.

Requests must contain an `Authorization` header with a `Bearer` token. This token should be a valid JWT token from Supabase (or another service that implements the same authentication protocol).

The auth handler then takes that token and verifies it with Supabase. If the token is valid, it returns the user's identity. If the token is invalid, it raises an exception. This means you must have a Supabase URL & key set in your environment variables to use this auth handler:

```bash
SUPABASE_URL=""
# Ensure this is your Supabase Service Role key
SUPABASE_KEY=""
```

The auth handler is then used as middleware for all requests to the server. It is configured to run on the following events:

* `threads.create`
* `threads.read`
* `threads.delete`
* `threads.update`
* `threads.search`
* `assistants.create`
* `assistants.read`
* `assistants.delete`
* `assistants.update`
* `assistants.search`
* `store`

For creation methods, it auto-injects the user's ID into the metadata. This is then uses in all read/update/delete/search methods to ensure that the user can only access their own threads and assistants.

By using custom authentication, we can call this LangGraph server directly from a frontend application, without having to worry about exposing API keys/secrets, since you only need a JWT token from Supabase to authenticate.

For more info, see our [LangGraph custom auth docs](https://langchain-ai.github.io/langgraph/tutorials/auth/getting_started/).
