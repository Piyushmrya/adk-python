import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool

class SamplingServer(Server):

    async def list_tools(self):
        return [
            Tool(
                name="trigger_sampling",
                description="Triggers sampling",
                inputSchema={"type": "object", "properties": {}},
            )
        ]

    async def call_tool(self, name, arguments):
        if name == "trigger_sampling":
            print("Server: requesting completion (sampling)...")

            response = await self.completion(
                prompt="Tell me something impressive.",
                temperature=0.6,
            )

            print("Server: received sampled response:", response)

            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Server got: {response}",
                    }
                ]
            }

        raise ValueError("Unknown tool")

async def main():
    server = SamplingServer(name="raw-sampling-server", version="1.0.0")

    async with stdio_server() as (read_stream, write_stream):
        init_opts = server.create_initialization_options()
        await server.run(read_stream, write_stream, init_opts)

if __name__ == "__main__":
    asyncio.run(main())