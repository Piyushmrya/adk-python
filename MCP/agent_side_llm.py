import os
import pathlib
import shutil
import certifi
import openai
import logging
import httpx
from google.adk.models.lite_llm import LiteLlm
from ..adk_agent_setup import adk_agent_config as config

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

_MERGED_CA = _OPENAI = _LITE = _SUMMARY_CLIENT = None

def init_ca_bundle(target_cert_file: str = "/usr/local/share/ca-certificates/zscaler.crt") -> str:
    """Merge Docker OS cert into certifi and set SSL envs."""
    global _MERGED_CA
    if _MERGED_CA:
        return _MERGED_CA
    # 1. Identify where to put the new merged file
    base_certifi = certifi.where()
    merged_path = str(pathlib.Path(__file__).parent / "merged_ca.pem")

    # 2. Check if the Zscaler cert from Docker exists
    if not os.path.exists(target_cert_file):
        logger.warning(f"Zscaler cert not found at {target_cert_file}. Using default certifi.")
        return base_certifi

    # 3. Create the merged file (Certifi + Zscaler)
    with open(base_certifi, "rb") as src, open(merged_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
        dst.write(b"\n") # newline separator
        with open(target_cert_file, "rb") as extra:
            dst.write(extra.read())

    # 4. Set Environment Variables (Global Fix)
    os.environ["SSL_CERT_FILE"] = merged_path
    os.environ["REQUESTS_CA_BUNDLE"] = merged_path
    os.environ["CURL_CA_BUNDLE"] = merged_path

    logger.info(f"SSL Bundle successfully created at: {merged_path}")
    _MERGED_CA = merged_path
    return merged_path


if config.LOCAL_RUN:
    logger.info("[SSL] LOCAL_RUN enabled, using custom SSL_CERT_FILE")
    CA_BUNDLE_PATH = init_ca_bundle(config.SSL_CERT_FILE)
else:
    logger.info("LOCAL_RUN disabled....")
    CA_BUNDLE_PATH = init_ca_bundle()


def get_headers(
        tenant_id: str = None,
        user_id: str = None,
        role_id: str = None,
        workspace_id: str = None
) -> dict:
    """Generate required headers. Raises ValueError if any required header is missing."""
    if not tenant_id:
        logging.info("TENANT_ID is missing")
        raise ValueError("TENANT_ID is required")
    if not user_id:
        logging.info("USER_ID is missing")
        raise ValueError("USER_ID is required")
    if not role_id:
        logging.info("ROLE_ID is missing")
        raise ValueError("ROLE_ID is required")
    if not workspace_id:
        logging.info("WORKSPACE is missing")
        raise ValueError("WORKSPACE_ID is required")

    return {
        "x-tenant-id": tenant_id,
        "x-user-id": user_id,
        "x-role-id": role_id,
        "x-workspace-id": workspace_id,
        "x-user-timezone": "Asia/Calcutta"
    }

def get_summary_client() -> openai.OpenAI:
    """Return the configured OpenAI client for short-term summary generation."""
    return LLMFactory.create_summary_client()


def get_summary_model() -> str:
    """Return the model name for short-term summary generation."""
    if config.AZURE_OPEN_AI_SETUP:
        return "gpt-4.1"
    return "gpt-4o-mini"


def get_lite_llm(model: str = "openai/gpt-5") -> LiteLlm:
    """Return LiteLLM configured for Azure or OpenAI."""
    return LLMFactory.create_lite_llm(model)


class LLMFactory:
    """Factory to create LLM clients based on runtime configuration."""

    @staticmethod
    def _get_httpx_client():
        """Returns client that explicitly trusts our merged bundle with retries."""
        transport = httpx.HTTPTransport(
            verify=CA_BUNDLE_PATH,
            retries=3
        )
        return httpx.Client(
            transport=transport,
            timeout=60.0,
            trust_env=True
        )

    @staticmethod
    def create_summary_client() -> openai.OpenAI:
        """Factory method for summary client."""
        global _SUMMARY_CLIENT

        if _SUMMARY_CLIENT:
            return _SUMMARY_CLIENT

        if config.AZURE_OPEN_AI_SETUP:
            _SUMMARY_CLIENT = LLMFactory._create_azure_summary_strategy()
            logger.info("LLM init with Azure Openai")
        else:
            _SUMMARY_CLIENT = LLMFactory._create_openai_strategy()
            logger.info("LLM init with Openai")

        return _SUMMARY_CLIENT

    @staticmethod
    def _create_azure_summary_strategy() -> openai.OpenAI:
        """Azure OpenAI strategy for summary client."""
        azure_endpoint = config.AZURE_OPENAI_ENDPOINT.rstrip('/')
        if not config.AZURE_OPENAI_API_KEY:
            raise RuntimeError("config.AZURE_OPENAI_API_KEY missing when Azure setup is enabled.")
        if not azure_endpoint:
            raise RuntimeError("AZURE_OPENAI_ENDPOINT missing or empty.")

        deployment = config.AZURE_OPENAI_DEPLOYMENT or "gpt-4.1"

        client = openai.OpenAI(
            base_url=f"{azure_endpoint}/openai/deployments/{deployment}",
            default_query={"api-version": config.AZURE_OPENAI_API_VERSION},
            default_headers={"api-key": config.AZURE_OPENAI_API_KEY},
            http_client=LLMFactory._get_httpx_client()
        )

        logger.info(
            f"[LLM] Factory: Azure OpenAI strategy - "
            f"endpoint={azure_endpoint}, version={config.AZURE_OPENAI_API_VERSION}"
        )
        return client

    @staticmethod
    def _create_openai_strategy() -> openai.OpenAI:
        """Classic OpenAI strategy."""
        global _OPENAI
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY missing.")
        _OPENAI = openai.OpenAI(
            api_key=config.OPENAI_API_KEY,
            http_client=LLMFactory._get_httpx_client()
        )
        logger.info("[LLM] Factory: OpenAI strategy - GPT-5 fallback")
        return _OPENAI

    @staticmethod
    def create_lite_llm(model: str = "openai/gpt-5") -> LiteLlm:
        """Factory method for LiteLLM."""
        global _LITE

        if _LITE:
            return _LITE

        if config.AZURE_OPEN_AI_SETUP:
            _LITE = LLMFactory._create_azure_litellm_strategy(model)
            logger.info("LLM init with Azure Openai")
        else:
            _LITE = LLMFactory._create_openai_litellm_strategy(model)
            logger.info("LLM init with Openai")

        return _LITE

    @staticmethod
    def _create_azure_litellm_strategy(model: str) -> LiteLlm:
        """Azure LiteLLM strategy."""
        azure_base_url = config.AZURE_OPENAI_ENDPOINT.rstrip('/')
        if not config.AZURE_OPENAI_API_KEY:
            raise RuntimeError("AZURE_OPENAI_API_KEY missing.")
        if not azure_base_url:
            raise RuntimeError("AZURE_OPENAI_ENDPOINT missing.")

        llm_model = f"azure/{config.AZURE_OPENAI_DEPLOYMENT}"
        litellm = LiteLlm(
            model=llm_model,
            api_key=config.AZURE_OPENAI_API_KEY,
            base_url=azure_base_url,
            api_version=config.AZURE_OPENAI_API_VERSION,
            reasoning_effort="low",
            seed=42,
            drop_params=True
        )
        logger.info(f"[LLM] Factory: Azure LiteLLM strategy - model={llm_model}, base_url={azure_base_url}")
        return litellm

    @staticmethod
    def _create_openai_litellm_strategy(model: str) -> LiteLlm:
        """OpenAI LiteLLM strategy."""
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY missing.")
        litellm = LiteLlm(model=model,
                          api_key=config.OPENAI_API_KEY,
                          reasoning_effort="low",
                          seed=42,
                          drop_params=True,
                          )
        logger.info(f"[LLM] Factory: OpenAI LiteLLM strategy - model={model}")
        return litellm



#Adkconfig
#
# /*
# import os
#
# LOCAL_RUN = os.getenv("LOCAL_RUN", True)
# STM_DIR = os.getenv("STM_DIR","/app/stm")
# GCP_MCP_LOCAL = os.getenv("GCP_MCP_LOCAL", False)
# if LOCAL_RUN:
#     from dotenv import load_dotenv
#     BASE_DIR = os.path.dirname(__file__)
#     ENV_PATH = os.path.join(BASE_DIR, ".env")
#     load_dotenv(dotenv_path=ENV_PATH)
#     SSL_CERT_FILE = os.getenv("SSL_CERT_FILE")
#
#     BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     STM_DIR = os.path.join(BASE_DIR, "conversations")
#
# AZURE_OPEN_AI_SETUP = os.getenv("AZURE_OPEN_AI_SETUP", False)
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") if not AZURE_OPEN_AI_SETUP else None
#
# MCP_SERVER_URL = os.getenv("MCP_SERVER_URL","https://demo.qa.powerme.cloud/api/composite-mcp")
# DESIRED_TOOL_NAMES = [
#     "pme_es_search",
#     "pme_es_get_mappings",
#     "pme_es_list_indices",
#     # "pme_asset_relations_pg",
#     # "pme_asset_relations_es",
#     # "pme_get_report_usage",
#     # "pme_get_system_asset_map",
#     # "pme_search_edm",
#     "pme_output_refinement",
#     # "pme_get_asset_expression",
#     "reasoning_tool"
# ]
# TENANT_ID = os.getenv("TENANT_ID")
# AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY") if AZURE_OPEN_AI_SETUP else None
# AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
# AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
# AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://genai.heineken.com/")
#
# */


