import asyncio
from fastmcp import FastMCP

mcp = FastMCP("sampling-test-server")

@mcp.tool()
async def trigger_sampling():
    print("Server: requesting sampling from client...")

    response = await mcp.sample(
        prompt="Respond with something cool",
        temperature=0.7,
    )

    print("Server: got sampled response:", response)

    return f"Server received: {response}"

if __name__ == "__main__":
    mcp.run()