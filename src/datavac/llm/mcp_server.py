"""A local (stdio) MCP server exposing the DataVacuum database tools, for use by an MCP client such as a coding agent.

Launch with `datavac mcp`. Requires the `mcp` extra (fastmcp).
"""
from __future__ import annotations
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastmcp import FastMCP


def build_server() -> FastMCP:
    from fastmcp import FastMCP
    from datavac.llm.llm_dbtools import DB_TOOLS, DB_TOOLS_GUIDANCE
    mcp = FastMCP("DataVacuum", instructions=DB_TOOLS_GUIDANCE)
    for func in DB_TOOLS: mcp.tool(func)
    return mcp


def cli_mcp(*args):
    import argparse
    parser = argparse.ArgumentParser(prog='mcp', description='Run a local stdio MCP server exposing the DataVacuum database tools')
    parser.add_argument('--log_level', '-l', default='INFO', help='Log level for datavac logging, which goes to stderr')
    namespace = parser.parse_args(args)

    # stdout carries the protocol, so make sure datavac logging never goes there
    from datavac.util import dvlogging
    dvlogging.set_log_stream(sys.stderr)
    dvlogging.set_level(namespace.log_level)

    build_server().run(transport='stdio', show_banner=False)


if __name__ == '__main__':
    cli_mcp(*sys.argv[1:])
