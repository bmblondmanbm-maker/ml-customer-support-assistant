"""Minimal internal tool-calling layer for the assistant."""

from __future__ import annotations

from typing import Any, Callable, Dict, Type

from pydantic import BaseModel, Field, ValidationError


class AccountStatusArgs(BaseModel):
    """Arguments for account-status lookup."""

    account_id: str = Field(min_length=3)


class ProductInfoArgs(BaseModel):
    """Arguments for product-information lookup."""

    product_name: str = Field(min_length=2)


def get_account_status(args: AccountStatusArgs) -> Dict[str, Any]:
    """Mock internal account-status tool."""

    return {
        'tool': 'get_account_status',
        'account_id': args.account_id,
        'status': 'active',
        'source': 'mock_internal_service',
    }


def get_product_info(args: ProductInfoArgs) -> Dict[str, Any]:
    """Mock internal product-information tool."""

    return {
        'tool': 'get_product_info',
        'product_name': args.product_name,
        'available': True,
        'source': 'mock_internal_catalog',
    }


TOOL_REGISTRY = {
    'get_account_status': {
        'schema': AccountStatusArgs,
        'function': get_account_status,
    },
    'get_product_info': {
        'schema': ProductInfoArgs,
        'function': get_product_info,
    },
}


def call_tool(
    tool_name: str,
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate arguments and execute a registered tool."""

    if tool_name not in TOOL_REGISTRY:
        return {
            'ok': False,
            'error': 'unknown_tool',
            'tool_name': tool_name,
        }

    tool = TOOL_REGISTRY[tool_name]
    schema: Type[BaseModel] = tool['schema']
    function: Callable = tool['function']

    try:
        validated_args = schema.model_validate(arguments)
    except ValidationError as exc:
        return {
            'ok': False,
            'error': 'invalid_arguments',
            'tool_name': tool_name,
            'details': exc.errors(),
        }

    result = function(validated_args)

    return {
        'ok': True,
        'tool_name': tool_name,
        'result': result,
    }


def get_tool_descriptions() -> Dict[str, Dict[str, Any]]:
    """Return lightweight metadata for available tools."""

    return {
        'get_account_status': {
            'description': 'Get account status by account_id.',
            'arguments': {
                'account_id': 'string',
            },
        },
        'get_product_info': {
            'description': 'Get product information by product name.',
            'arguments': {
                'product_name': 'string',
            },
        },
    }
