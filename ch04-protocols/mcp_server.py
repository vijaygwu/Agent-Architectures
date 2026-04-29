"""
Chapter 4: MCP Server Implementation
=====================================
A complete Model Context Protocol server for enterprise tool integration.

This example implements an MCP server that exposes procurement tools
to AI applications following the official MCP specification.

Reference: https://modelcontextprotocol.io/specification/2025-11-25
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Sequence
from dataclasses import dataclass, asdict

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel,
    Resource,
    ResourceTemplate,
    Prompt,
    PromptMessage,
    PromptArgument,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("procurement-mcp-server")


# =============================================================================
# Domain Models
# =============================================================================

@dataclass
class Vendor:
    id: str
    name: str
    category: str
    risk_score: float
    contact_email: str
    payment_terms: int  # days
    active: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PurchaseOrder:
    id: str
    vendor_id: str
    items: list[dict]
    total_amount: float
    currency: str
    status: str  # draft, pending_approval, approved, rejected, completed
    created_by: str
    created_at: str
    approver: str | None = None
    approved_at: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# Mock Database (Replace with real database in production)
# =============================================================================

class ProcurementDatabase:
    """Simulated procurement database for demonstration"""

    def __init__(self):
        self.vendors: dict[str, Vendor] = {
            "V001": Vendor(
                id="V001",
                name="Acme Supplies Co",
                category="office_supplies",
                risk_score=0.15,
                contact_email="sales@acme.com",
                payment_terms=30,
                active=True
            ),
            "V002": Vendor(
                id="V002",
                name="TechParts International",
                category="electronics",
                risk_score=0.25,
                contact_email="orders@techparts.com",
                payment_terms=45,
                active=True
            ),
            "V003": Vendor(
                id="V003",
                name="Global Logistics Ltd",
                category="shipping",
                risk_score=0.10,
                contact_email="shipping@globallog.com",
                payment_terms=15,
                active=True
            ),
        }

        self.purchase_orders: dict[str, PurchaseOrder] = {}
        self.po_counter = 1000

    async def get_vendor(self, vendor_id: str) -> Vendor | None:
        return self.vendors.get(vendor_id)

    async def search_vendors(
        self,
        category: str | None = None,
        max_risk_score: float | None = None
    ) -> list[Vendor]:
        results = list(self.vendors.values())

        if category:
            results = [v for v in results if v.category == category]

        if max_risk_score is not None:
            results = [v for v in results if v.risk_score <= max_risk_score]

        return results

    async def create_purchase_order(
        self,
        vendor_id: str,
        items: list[dict],
        created_by: str,
        approver: str
    ) -> PurchaseOrder:
        self.po_counter += 1
        po_id = f"PO{self.po_counter}"

        total = sum(item.get("quantity", 1) * item.get("unit_price", 0) for item in items)

        po = PurchaseOrder(
            id=po_id,
            vendor_id=vendor_id,
            items=items,
            total_amount=total,
            currency="USD",
            status="pending_approval",
            created_by=created_by,
            created_at=datetime.utcnow().isoformat(),
            approver=approver
        )

        self.purchase_orders[po_id] = po
        return po

    async def get_purchase_order(self, po_id: str) -> PurchaseOrder | None:
        return self.purchase_orders.get(po_id)

    async def update_po_status(self, po_id: str, status: str) -> PurchaseOrder | None:
        po = self.purchase_orders.get(po_id)
        if po:
            po.status = status
            if status == "approved":
                po.approved_at = datetime.utcnow().isoformat()
        return po


# =============================================================================
# MCP Server Implementation
# =============================================================================

# Initialize database
db = ProcurementDatabase()

# Create MCP server
server = Server("procurement-mcp-server")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """
    List all available tools.

    Tools are model-controlled - the LLM decides when to invoke them
    based on the user's request and tool descriptions.
    """
    return [
        Tool(
            name="lookup_vendor",
            description=(
                "Look up detailed information about a specific vendor by their ID. "
                "Returns vendor details including name, category, risk score, "
                "contact information, and payment terms."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "vendor_id": {
                        "type": "string",
                        "description": "The unique vendor identifier (e.g., V001)"
                    }
                },
                "required": ["vendor_id"]
            }
        ),
        Tool(
            name="search_vendors",
            description=(
                "Search for vendors matching specified criteria. "
                "Can filter by category and maximum acceptable risk score."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Vendor category to filter by",
                        "enum": ["office_supplies", "electronics", "shipping", "raw_materials"]
                    },
                    "max_risk_score": {
                        "type": "number",
                        "description": "Maximum acceptable risk score (0.0 to 1.0)",
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                }
            }
        ),
        Tool(
            name="create_purchase_order",
            description=(
                "Create a new purchase order for a vendor. "
                "The PO will be created in 'pending_approval' status and "
                "routed to the specified approver. Returns the created PO details."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "vendor_id": {
                        "type": "string",
                        "description": "The vendor to create the PO for"
                    },
                    "items": {
                        "type": "array",
                        "description": "List of items to order",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string"},
                                "quantity": {"type": "integer", "minimum": 1},
                                "unit_price": {"type": "number", "minimum": 0}
                            },
                            "required": ["description", "quantity", "unit_price"]
                        }
                    },
                    "approver": {
                        "type": "string",
                        "description": "Email of the person who should approve this PO"
                    }
                },
                "required": ["vendor_id", "items", "approver"]
            }
        ),
        Tool(
            name="get_purchase_order",
            description="Retrieve details of an existing purchase order by its ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "po_id": {
                        "type": "string",
                        "description": "The purchase order ID (e.g., PO1001)"
                    }
                },
                "required": ["po_id"]
            }
        ),
        Tool(
            name="approve_purchase_order",
            description=(
                "Approve a pending purchase order. "
                "Only POs in 'pending_approval' status can be approved."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "po_id": {
                        "type": "string",
                        "description": "The purchase order ID to approve"
                    }
                },
                "required": ["po_id"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> Sequence[TextContent]:
    """
    Handle tool invocations.

    Each tool call is logged for audit purposes and returns structured
    JSON responses that the LLM can interpret.
    """
    logger.info(f"Tool called: {name} with arguments: {arguments}")

    try:
        if name == "lookup_vendor":
            vendor = await db.get_vendor(arguments["vendor_id"])
            if vendor:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": True,
                        "vendor": vendor.to_dict()
                    }, indent=2)
                )]
            else:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": False,
                        "error": f"Vendor {arguments['vendor_id']} not found"
                    })
                )]

        elif name == "search_vendors":
            vendors = await db.search_vendors(
                category=arguments.get("category"),
                max_risk_score=arguments.get("max_risk_score")
            )
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "count": len(vendors),
                    "vendors": [v.to_dict() for v in vendors]
                }, indent=2)
            )]

        elif name == "create_purchase_order":
            # Validate vendor exists
            vendor = await db.get_vendor(arguments["vendor_id"])
            if not vendor:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": False,
                        "error": f"Vendor {arguments['vendor_id']} not found"
                    })
                )]

            if not vendor.active:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": False,
                        "error": f"Vendor {vendor.name} is not active"
                    })
                )]

            po = await db.create_purchase_order(
                vendor_id=arguments["vendor_id"],
                items=arguments["items"],
                created_by="mcp-agent",  # In production, get from auth context
                approver=arguments["approver"]
            )

            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "message": f"Purchase order {po.id} created successfully",
                    "purchase_order": po.to_dict()
                }, indent=2)
            )]

        elif name == "get_purchase_order":
            po = await db.get_purchase_order(arguments["po_id"])
            if po:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": True,
                        "purchase_order": po.to_dict()
                    }, indent=2)
                )]
            else:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": False,
                        "error": f"Purchase order {arguments['po_id']} not found"
                    })
                )]

        elif name == "approve_purchase_order":
            po = await db.get_purchase_order(arguments["po_id"])
            if not po:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": False,
                        "error": f"Purchase order {arguments['po_id']} not found"
                    })
                )]

            if po.status != "pending_approval":
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": False,
                        "error": f"Cannot approve PO in status '{po.status}'"
                    })
                )]

            updated_po = await db.update_po_status(arguments["po_id"], "approved")
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "message": f"Purchase order {po.id} approved",
                    "purchase_order": updated_po.to_dict()
                }, indent=2)
            )]

        else:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": False,
                    "error": f"Unknown tool: {name}"
                })
            )]

    except Exception as e:
        logger.exception(f"Error in tool {name}")
        return [TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "error": str(e),
                "isError": True
            })
        )]


@server.list_resources()
async def list_resources() -> list[Resource]:
    """
    List available resources.

    Resources are application-controlled - the host application decides
    when to include them in context.
    """
    return [
        Resource(
            uri="procurement://vendors/catalog",
            name="Vendor Catalog",
            description="Complete catalog of approved vendors",
            mimeType="application/json"
        ),
        Resource(
            uri="procurement://policies/approval-matrix",
            name="Approval Matrix",
            description="PO approval thresholds and routing rules",
            mimeType="application/json"
        )
    ]


@server.read_resource()
async def read_resource(uri: str) -> str:
    """Read resource content by URI"""

    if uri == "procurement://vendors/catalog":
        vendors = await db.search_vendors()
        return json.dumps({
            "catalog_version": "2024.1",
            "last_updated": datetime.utcnow().isoformat(),
            "vendors": [v.to_dict() for v in vendors]
        }, indent=2)

    elif uri == "procurement://policies/approval-matrix":
        return json.dumps({
            "approval_matrix": [
                {"max_amount": 1000, "approver_level": "team_lead"},
                {"max_amount": 10000, "approver_level": "manager"},
                {"max_amount": 50000, "approver_level": "director"},
                {"max_amount": float("inf"), "approver_level": "vp"}
            ],
            "special_categories": {
                "electronics": {"additional_approval": "it_security"},
                "software": {"additional_approval": "it_architecture"}
            }
        }, indent=2)

    raise ValueError(f"Unknown resource: {uri}")


@server.list_prompts()
async def list_prompts() -> list[Prompt]:
    """
    List available prompt templates.

    Prompts are user-controlled templates that help users interact
    with the procurement system effectively.
    """
    return [
        Prompt(
            name="find_vendor",
            description="Find a vendor for a specific procurement need",
            arguments=[
                PromptArgument(
                    name="need",
                    description="What you need to procure",
                    required=True
                ),
                PromptArgument(
                    name="budget",
                    description="Approximate budget",
                    required=False
                )
            ]
        ),
        Prompt(
            name="create_po_workflow",
            description="Step-by-step workflow for creating a purchase order",
            arguments=[
                PromptArgument(
                    name="vendor_id",
                    description="The vendor to order from",
                    required=True
                )
            ]
        )
    ]


@server.get_prompt()
async def get_prompt(name: str, arguments: dict[str, str] | None) -> list[PromptMessage]:
    """Get a prompt template with filled arguments"""

    if name == "find_vendor":
        need = arguments.get("need", "general supplies") if arguments else "general supplies"
        budget = arguments.get("budget", "not specified") if arguments else "not specified"

        return [
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=f"""I need to find a vendor for the following procurement need:

Need: {need}
Budget: {budget}

Please:
1. Search for vendors that might fulfill this need
2. Compare their risk scores and payment terms
3. Recommend the best option with justification"""
                )
            )
        ]

    elif name == "create_po_workflow":
        vendor_id = arguments.get("vendor_id") if arguments else None
        if not vendor_id:
            raise ValueError("vendor_id is required")

        return [
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=f"""Help me create a purchase order for vendor {vendor_id}.

Please guide me through:
1. First, look up the vendor details to confirm they're active
2. Help me specify the items I want to order
3. Create the purchase order with appropriate approval routing
4. Summarize the created PO and next steps"""
                )
            )
        ]

    raise ValueError(f"Unknown prompt: {name}")


# =============================================================================
# Server Entry Point
# =============================================================================

async def main():
    """Run the MCP server using stdio transport"""
    logger.info("Starting Procurement MCP Server")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
