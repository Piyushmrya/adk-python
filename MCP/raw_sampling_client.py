import asyncio
from datetime import timedelta
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

async def sampling_callback(request):
    print("🔥 SAMPLING CALLBACK TRIGGERED")
    print("Prompt from server:", request.prompt)
    return "Client response from sampling callback."

async def main():
    client = stdio_client(
        server=StdioServerParameters(
            command="python",
            args=["raw_sampling_server.py"],
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