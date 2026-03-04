import asyncio
from mcp.client.stdio import stdio_client
from mcp import ClientSession
from mcp import StdioServerParameters
from datetime import timedelta

async def sampling_callback(request):
    print("🔥 SAMPLING CALLBACK TRIGGERED")
    print("Prompt:", request.prompt)
    return "Client says hello 😎"

async def main():
    client = stdio_client(
        server=StdioServerParameters(
            command="python",
            args=["fastmcp_sampling_server.py"],
        )
    )

    async with client as transports:
        async with ClientSession(
                *transports[:2],
                read_timeout_seconds=timedelta(seconds=15),
                sampling_callback=sampling_callback,
        ) as session:

            await session.initialize()

            tools = await session.list_tools()
            print("Tools:", [t.name for t in tools.tools])

            result = await session.call_tool(
                name="trigger_sampling",
                arguments={}
            )

            print("Final tool result:", result)

asyncio.run(main())