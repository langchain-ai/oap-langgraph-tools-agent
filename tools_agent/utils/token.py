import logging
import aiohttp
from typing import Dict, Optional, Any
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_store


async def get_mcp_access_token(
    supabase_token: str,
    base_mcp_url: str,
) -> Optional[Dict[str, Any]]:
    """
    Exchange a Supabase token for an MCP access token.

    Args:
        supabase_token: The Supabase token to exchange
        base_mcp_url: The base URL for the MCP server

    Returns:
        The token data as a dictionary if successful, None otherwise
    """
    try:
        # Exchange Supabase token for MCP access token
        form_data = {
            "client_id": "mcp_default",
            "subject_token": supabase_token,
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "resource": base_mcp_url.rstrip("/") + "/mcp",
            "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                base_mcp_url.rstrip("/") + "/oauth/token",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data=form_data,
            ) as token_response:
                if token_response.status == 200:
                    token_data = await token_response.json()
                    return token_data
                else:
                    response_text = await token_response.text()
                    logging.error(f"Token exchange failed: {response_text}")
    except Exception as e:
        logging.error(f"Error during token exchange: {e}")

    return None


async def get_tokens(config: RunnableConfig):
    store = get_store()
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return None

    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return None

    tokens = await store.aget((user_id, "tokens"), "data")
    if not tokens:
        return None

    expires_in = tokens.value.get("expires_in")  # seconds until expiration
    created_at = tokens.created_at  # datetime of token creation

    from datetime import datetime, timedelta, timezone

    current_time = datetime.now(timezone.utc)
    expiration_time = created_at + timedelta(seconds=expires_in)

    if current_time > expiration_time:
        # Tokens have expired, delete them
        await store.adelete((user_id, "tokens"), "data")
        return None

    return tokens.value


async def set_tokens(config: RunnableConfig, tokens: dict[str, Any]):
    store = get_store()
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return

    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return

    await store.aput((user_id, "tokens"), "data", tokens)
    return


async def fetch_tokens(config: RunnableConfig) -> dict[str, Any] | None:
    """
    Fetch MCP access token if it doesn't already exist in the store.

    Args:
        config: The runnable configuration

    Raises:
        ValueError: If required configuration is missing
    """

    current_tokens = await get_tokens(config)
    if current_tokens:
        return current_tokens

    supabase_token = config.get("configurable", {}).get("x-supabase-access-token")
    if not supabase_token:
        return None

    mcp_config = config.get("configurable", {}).get("mcp_config")
    if not mcp_config or not mcp_config.get("url"):
        return None

    mcp_tokens = await get_mcp_access_token(supabase_token, mcp_config.get("url"))

    await set_tokens(config, mcp_tokens)
    return mcp_tokens
