from typing import Annotated
from langchain_core.tools import StructuredTool, ToolException, tool
import aiohttp
import re
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession, Tool, McpError


def create_langchain_mcp_tool(
    mcp_tool: Tool, mcp_server_url: str, mcp_access_token: str
) -> StructuredTool:
    """Create a LangChain tool from an MCP tool."""

    @tool(
        mcp_tool.name,
        description=mcp_tool.description,
        args_schema=mcp_tool.inputSchema,
    )
    async def new_tool(**kwargs):
        async with streamablehttp_client(
            mcp_server_url, headers={"Authorization": f"Bearer {mcp_access_token}"}
        ) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as tool_session:
                await tool_session.initialize()
                return await tool_session.call_tool(mcp_tool.name, arguments=kwargs)

    return new_tool


def wrap_mcp_authenticate_tool(tool: StructuredTool) -> StructuredTool:
    """Wrap the tool coroutine to handle `interaction_required` MCP error.

    Tried to obtain the URL from the error, which the LLM can use to render a link."""

    old_coroutine = tool.coroutine

    async def wrapped_mcp_coroutine(**kwargs):
        def _find_first_mcp_error_nested(exc: BaseException) -> McpError | None:
            if isinstance(exc, McpError):  # McpError needs to be in scope
                return exc
            if isinstance(exc, ExceptionGroup):
                for sub_exc in exc.exceptions:
                    found = _find_first_mcp_error_nested(sub_exc)
                    if found:
                        return found
            return None

        try:
            response = await old_coroutine(**kwargs)
            return response
        except BaseException as e_orig:
            mcp_error_instance = _find_first_mcp_error_nested(e_orig)

            if mcp_error_instance:
                current_error_details = mcp_error_instance.error
                if (
                    hasattr(current_error_details, "code")
                    and current_error_details.code == -32003  # interaction_required
                    and hasattr(current_error_details, "data")
                ):
                    error_data = current_error_details.data or {}
                    message_payload = error_data.get("message") or {}
                    error_message_text = (
                        message_payload.get("text")
                        if isinstance(message_payload, dict)
                        else "Required interaction"
                    )
                    error_message_text = error_message_text or "Required interaction"

                    if url := error_data.get("url"):
                        error_message_text += f" {url}"
                    raise ToolException(error_message_text) from e_orig
                else:
                    raise e_orig

    tool.coroutine = wrapped_mcp_coroutine
    return tool


async def create_rag_tool(rag_url: str, collection_id: str, access_token: str):
    """Create a RAG tool for a specific collection.

    Args:
        rag_url: The base URL for the RAG API server
        collection_id: The ID of the collection to query
        access_token: The access token for authentication

    Returns:
        A structured tool that can be used to query the RAG collection
    """
    if rag_url.endswith("/"):
        rag_url = rag_url[:-1]

    collection_endpoint = f"{rag_url}/collections/{collection_id}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                collection_endpoint, headers={"Authorization": f"Bearer {access_token}"}
            ) as response:
                response.raise_for_status()
                collection_data = await response.json()

        # Get the collection name and sanitize it to match the required regex pattern
        raw_collection_name = collection_data.get("name", f"collection_{collection_id}")

        # Sanitize the name to only include alphanumeric characters, underscores, and hyphens
        # Replace any other characters with underscores
        sanitized_name = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_collection_name)

        # Ensure the name is not empty and doesn't exceed 64 characters
        if not sanitized_name:
            sanitized_name = f"collection_{collection_id}"
        collection_name = sanitized_name[:64]

        raw_description = collection_data.get("metadata", {}).get("description")

        if not raw_description:
            collection_description = "Search your collection of documents for results semantically similar to the input query"
        else:
            collection_description = f"Search your collection of documents for results semantically similar to the input query. Collection description: {raw_description}"

        @tool(name_or_callable=collection_name, description=collection_description)
        async def get_documents(
            query: Annotated[str, "The search query to find relevant documents"],
        ) -> str:
            """Search for documents in the collection based on the query"""

            search_endpoint = f"{rag_url}/collections/{collection_id}/documents/search"
            payload = {"query": query, "limit": 10}

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        search_endpoint,
                        json=payload,
                        headers={"Authorization": f"Bearer {access_token}"},
                    ) as search_response:
                        search_response.raise_for_status()
                        documents = await search_response.json()

                formatted_docs = "<all-documents>\n"

                for doc in documents:
                    doc_id = doc.get("id", "unknown")
                    content = doc.get("page_content", "")
                    formatted_docs += (
                        f'  <document id="{doc_id}">\n    {content}\n  </document>\n'
                    )

                formatted_docs += "</all-documents>"
                return formatted_docs
            except Exception as e:
                return f"<all-documents>\n  <error>{str(e)}</error>\n</all-documents>"

        return get_documents

    except Exception as e:
        raise Exception(f"Failed to create RAG tool: {str(e)}")
