# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Configuration management for web interface.
Handles:
- Namespace discovery
- CLI argument creation
- Model configuration
- Agent config setup
"""
import os
from argparse import Namespace
from functools import lru_cache
from typing import Any, Dict, List,Tuple

import structlog

from datus.cli.repl import DatusCLI
from datus.configuration.agent_config_loader import parse_config_path
from datus.models.base import LLMBaseModel

logger = structlog.get_logger("web_chatbot.config")


def get_available_namespaces(config_path: str = "") -> List[str]:
    """Extract available namespaces from config file"""
    try:
        config = _load_config_cached(config_path)
        if "agent" in config and "namespace" in config["agent"]:
            return list(config["agent"]["namespace"].keys())
        elif "namespace" in config:
            return list(config["namespace"].keys())
        return []
    except Exception as e:
        logger.error(f"Failed to read namespaces from config: {e}")
        return []


def create_cli_args(config_path: str = "", namespace: str = None, catalog: str = "", database: str = "") -> Namespace:
    """Create CLI arguments for DatusCLI initialization"""
    # Import here to avoid circular dependency with streamlit session_state
    import streamlit as st

    args = Namespace()
    args.config = parse_config_path(config_path)
    args.namespace = namespace  # Add namespace parameter
    args.history_file = ".datus_history"
    args.db_type = "sqlite"
    args.db_path = None
    args.database = database
    args.catalog = catalog
    args.schema = ""
    # Add missing attributes that DatusCLI expects
    args.debug = bool(st.session_state.get("startup_debug", False)) if hasattr(st, "session_state") else False
    args.no_color = False

    # Read storage path from config file
    args.storage_path = get_storage_path_from_config(config_path)

    args.save_llm_trace = False
    # Add non-interactive mode flags
    args.non_interactive = True
    args.disable_detail_views = True
    return args


@lru_cache(maxsize=1)
def _load_config_cached(config_path: str) -> Dict[str, Any]:
    """Load and cache YAML configuration"""
    import yaml

    config_path = parse_config_path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_storage_path_from_config(config_path: str) -> str:
    """
    Get storage path from configuration.
    """
    try:
        config = _load_config_cached(config_path)
        # Get home from config, default to ~/.datus
        home = config.get("agent", {}).get("home", "~/.datus")
        from datus.utils.path_manager import get_path_manager

        # Update path manager with configured home and return data_dir
        pm = get_path_manager()
        pm.update_home(home)
        return str(pm.data_dir)
    except Exception as e:
        logger.warning(f"Failed to read storage path from config: {e}")
        # Fallback to default
        return os.path.expanduser("~/.datus/data")


class ConfigManager:
    """Manages agent configuration and model settings."""

    def __init__(self, cli: DatusCLI = None):
        """
        Initialize ConfigManager.

        Args:
            cli: Optional DatusCLI instance. Can be updated later.
        """
        self.cli = cli

    def setup_config(
        self, config_path: str = "conf/agent.yml", namespace: str = None, catalog: str = "", database: str = ""
    ) -> DatusCLI:
        """
        Setup agent configuration by initializing real DatusCLI.

        Args:
            config_path: Path to agent configuration file
            namespace: Namespace to use (optional)
            catalog: Catalog to use (optional)
            database: Database to use (optional)

        Returns:
            Initialized DatusCLI instance

        Raises:
            Exception: If configuration loading fails
        """
        # Create CLI arguments
        args = create_cli_args(config_path, namespace, catalog, database=database)

        # Initialize real DatusCLI
        cli = DatusCLI(args)

        # Set Streamlit mode flag to skip interactive prompts
        cli.streamlit_mode = True

        # Update internal reference
        self.cli = cli

        return cli

    def get_available_models(self) -> List[str]:
        """Get list of available model names"""
        if not self.cli or not hasattr(self.cli.agent_config, "models"):
            return []

        try:
            return list(self.cli.agent_config.models.keys())
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return []

    def get_current_chat_model(self, config_path: str = "conf/agent.yml") -> str:
        """
        Get current chat model from configuration.

        Args:
            config_path: Path to configuration file

        Returns:
            Model name or "unknown"
        """
        try:
            config = _load_config_cached(config_path)
            chat_model = config.get("agent", {}).get("nodes", {}).get("chat", {}).get("model", "")
            if chat_model:
                return chat_model
            available_models = self.get_available_models()
            return available_models[0] if available_models else "unknown"
        except Exception as e:
            logger.error(f"Failed to get current chat model: {e}")
            return "unknown"

    def switch_chat_model(self, new_model_name: str) -> bool:
        """
        切换聊天模型

        Args:
            new_model_name: 新模型名称

        Returns:
            bool: 成功返回True，失败返回False
        """
        try:

            # 获取聊天节点
            if not hasattr(self.cli, 'chat_commands') or not self.cli.chat_commands:
                logger.error("聊天命令未初始化")
                return False

            chat_node = self.cli.chat_commands.chat_node
            if not chat_node:
                # 如果没有聊天节点，则初始化一个新的聊天节点
                # 获取当前选择的子代理名称，如果没有则使用None创建默认聊天节点
                subagent_name = self.cli.chat_commands.current_subagent_name
                new_node = self.cli.chat_commands._create_new_node(subagent_name=subagent_name)
                # 更新所有相关引用
                self.cli.chat_commands.current_node = new_node
                self.cli.chat_commands.chat_node = new_node
                chat_node = new_node

            # 切换模型
            chat_node.model = new_model_name
            # 初始化模型
            chat_node._initialize()
            logger.info(f"成功切换聊天模型为: {new_model_name}")
            return True

        except Exception as e:
            logger.error(f"切换聊天模型时发生错误: {e}")
            return False

    def validate_model_switch(self, model_name: str) -> Tuple[bool, str]:
        """
        验证模型切换是否可行

        Args:
            model_name: 要切换的模型名称

        Returns:
            Tuple[bool, str]: (是否有效, 错误消息)
        """
        # 检查模型是否存在
        if not self.cli or not hasattr(self.cli, 'agent_config'):
            return False, "CLI未初始化"

        if model_name not in self.cli.agent_config.models:
            available = ", ".join(self.cli.agent_config.models.keys())
            return False, f"未找到模型'{model_name}'。可用模型: {available}"

        # 检查聊天节点是否存在
        if not hasattr(self.cli, 'chat_commands') or not self.cli.chat_commands:
            return False, "聊天命令未初始化"

        chat_node = self.cli.chat_commands.chat_node
        if not chat_node:
            return False, "聊天节点未初始化"

        # 检查模型是否可以初始化（API密钥、连接等）
        try:
            test_model = LLMBaseModel.create_model(
                model_name=model_name,
                agent_config=self.cli.agent_config
            )
            return True, ""
        except Exception as e:
            return False, f"模型验证失败: {str(e)}"