from langchain_core.runnables import RunnableConfig
from typing import Optional, List
from pydantic import BaseModel, Field
from langgraph.prebuilt import create_react_agent
from tools_agent.utils.tools import create_rag_tool
from langchain.chat_models import init_chat_model
from tools_agent.utils.token import fetch_tokens
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession
from langchain_core.tools import StructuredTool
from tools_agent.utils.tools import (
    wrap_mcp_authenticate_tool,
    create_langchain_mcp_tool,
)


UNEDITABLE_SYSTEM_PROMPT = "\nIf the tool throws an error requiring authentication, provide the user with a Markdown link to the authentication page and prompt them to authenticate."

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that has access to a variety of tools."
)


class RagConfig(BaseModel):
    rag_url: Optional[str] = None
    """The URL of the rag server"""
    collections: Optional[List[str]] = None
    """The collections to use for rag"""


class MCPConfig(BaseModel):
    url: Optional[str] = Field(
        default=None,
        optional=True,
    )
    """The URL of the MCP server"""
    tools: Optional[List[str]] = Field(
        default=None,
        optional=True,
    )
    """The tools to make available to the LLM"""


class GraphConfigPydantic(BaseModel):
    model_name: Optional[str] = Field(
        default="anthropic:claude-3-7-sonnet-latest",
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "anthropic:claude-3-7-sonnet-latest",
                "description": "The model to use in all generations",
                "options": [
                    {
                        "label": "Claude 3.7 Sonnet",
                        "value": "anthropic:claude-3-7-sonnet-latest",
                    },
                    {
                        "label": "Claude 3.5 Sonnet",
                        "value": "anthropic:claude-3-5-sonnet-latest",
                    },
                    {"label": "GPT 4o", "value": "openai:gpt-4o"},
                    {"label": "GPT 4o mini", "value": "openai:gpt-4o-mini"},
                    {"label": "GPT 4.1", "value": "openai:gpt-4.1"},
                ],
            }
        },
    )
    temperature: Optional[float] = Field(
        default=0.7,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 0.7,
                "min": 0,
                "max": 2,
                "step": 0.1,
                "description": "Controls randomness (0 = deterministic, 2 = creative)",
            }
        },
    )
    max_tokens: Optional[int] = Field(
        default=4000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 4000,
                "min": 1,
                "description": "The maximum number of tokens to generate",
            }
        },
    )
    system_prompt: Optional[str] = Field(
        default=DEFAULT_SYSTEM_PROMPT,
        metadata={
            "x_oap_ui_config": {
                "type": "textarea",
                "placeholder": "Enter a system prompt...",
                "description": f"The system prompt to use in all generations. The following prompt will always be included at the end of the system prompt:\n---{UNEDITABLE_SYSTEM_PROMPT}\n---",
                "default": DEFAULT_SYSTEM_PROMPT,
            }
        },
    )
    mcp_config: Optional[MCPConfig] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "mcp",
                # Here is where you would set the default tools.
                # "default": {
                #     "tools": ["Math_Divide", "Math_Mod"]
                # }
            }
        },
    )
    rag: Optional[RagConfig] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "rag",
                # Here is where you would set the default collection. Use collection IDs
                # "default": {
                #     "collections": [
                #         "fd4fac19-886c-4ac8-8a59-fff37d2b847f",
                #         "659abb76-fdeb-428a-ac8f-03b111183e25",
                #     ]
                # },
            }
        },
    )


async def graph(config: RunnableConfig):
    cfg = GraphConfigPydantic(**config.get("configurable", {}))
    tools = []

    supabase_token = config.get("configurable", {}).get("x-supabase-access-token")
    if cfg.rag and cfg.rag.rag_url and cfg.rag.collections and supabase_token:
        for collection in cfg.rag.collections:
            rag_tool = await create_rag_tool(
                cfg.rag.rag_url, collection, supabase_token
            )
            tools.append(rag_tool)

    if (
        cfg.mcp_config
        and cfg.mcp_config.url
        and cfg.mcp_config.tools
        and (mcp_tokens := await fetch_tokens(config))
    ):
        server_url = cfg.mcp_config.url.rstrip("/") + "/mcp"
        access_token = mcp_tokens["access_token"]
        # Set of tool names we need to find from the configuration
        tool_names_to_find = set(cfg.mcp_config.tools)
        # List to store the actual tool objects that are found and match the request
        fetched_mcp_tools_list: list[StructuredTool] = []
        # Set to keep track of the names of tools already added
        names_of_tools_added = set()

        current_tools_cursor: str | None = None

        async with streamablehttp_client(
            server_url, headers={"Authorization": f"Bearer {access_token}"}
        ) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                while True:
                    tool_page_response = await session.list_tools(
                        cursor=current_tools_cursor
                    )

                    if tool_page_response and len(tool_page_response.tools) > 0:
                        for mcp_tool in tool_page_response.tools:
                            if (
                                mcp_tool.name in tool_names_to_find
                                and mcp_tool.name not in names_of_tools_added
                            ):
                                langchain_mcp_tool = create_langchain_mcp_tool(
                                    mcp_tool, server_url, access_token
                                )
                                fetched_mcp_tools_list.append(
                                    wrap_mcp_authenticate_tool(langchain_mcp_tool)
                                )
                                names_of_tools_added.add(mcp_tool.name)

                    current_tools_cursor = tool_page_response.nextCursor

                    all_requested_tools_found = False
                    if not tool_names_to_find:  # If cfg.mcp_config.tools was empty
                        all_requested_tools_found = (
                            True  # Condition met: zero out of zero tools found.
                        )
                    elif len(names_of_tools_added) == len(tool_names_to_find):
                        all_requested_tools_found = True

                    if not current_tools_cursor or all_requested_tools_found:
                        break

                tools.extend(fetched_mcp_tools_list)

    model = init_chat_model(
        cfg.model_name,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )

    return create_react_agent(
        prompt=cfg.system_prompt + UNEDITABLE_SYSTEM_PROMPT,
        model=model,
        tools=tools,
        config_schema=GraphConfigPydantic,
    )
