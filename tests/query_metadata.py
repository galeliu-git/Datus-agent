#!/usr/bin/env python3
"""
查看已构建的元数据
用法: python query_metadata.py --namespace test
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from datus.configuration.agent_config_loader import load_agent_config
from datus.storage.schema_metadata.store import SchemaWithValueRAG


def query_metadata(namespace: str, database_name: str = "", table_name: str = ""):
    """查询元数据"""

    # 加载配置
    config_path = Path(__file__).parent.parent / "conf" / "agent.yml"
    agent_config = load_agent_config(
        config=str(config_path)
    )
    # 设置命名空间
    agent_config.current_namespace = namespace

    # 创建 RAG 实例
    rag = SchemaWithValueRAG(agent_config)

    print("=" * 80)
    print(f"元数据查询 - 命名空间: {namespace}")
    print("=" * 80)

    # 1. 显示总体统计
    print("\n📊 总体统计:")
    print("-" * 80)
    schema_size = rag.get_schema_size()
    value_size = rag.get_value_size()
    print(f"  模式数量: {schema_size}")
    print(f"  样本值数量: {value_size}")

    # 2. 显示所有表
    print("\n📋 所有表列表:")
    print("-" * 80)
    all_schemas = rag.search_all_schemas(database_name=database_name)
    print(f"找到 {len(all_schemas)} 条记录")
    print("\n表详细信息:")

    # 显示每张表的详细信息
    if all_schemas.num_rows > 0:
        # 将表转换为字典格式
        schema_dict = all_schemas.to_pydict()

        for i in range(all_schemas.num_rows):
            print(f"\n  [{i+1}] 表: {schema_dict['table_name'][i]}")
            print(f"      数据库: {schema_dict['database_name'][i]}")
            print(f"      模式: {schema_dict['schema_name'][i]}")
            print(f"      类型: {schema_dict['table_type'][i]}")
            print(f"      标识符: {schema_dict['identifier'][i]}")

            # 显示列定义
            if 'definition' in schema_dict:
                definition = schema_dict['definition'][i]
                if definition:
                    # 只显示前200个字符
                    display_def = definition[:200] + "..." if len(definition) > 200 else definition
                    print(f"      定义: {display_def}")
    else:
        print("  没有找到任何表")

    # 3. 如果指定了表名，显示详细信息
    if table_name:
        print(f"\n🔍 表 '{table_name}' 的详细信息:")
        print("-" * 80)
        try:
            schemas, values = rag.search_tables([table_name], database_name=database_name)
            print(f"找到 {len(schemas)} 个匹配的模式")
            for schema in schemas:
                print(f"\n  表名: {schema.table_name}")
                print(f"  数据库: {schema.database_name}")
                print(f"  模式: {schema.schema_name}")
                print(f"  类型: {schema.table_type}")
                print(f"  定义:\n{schema.definition}")

            if values:
                print(f"\n  样本数据:")
                for value in values:
                    print(f"    - {value.sample_rows[:100]}..." if len(value.sample_rows) > 100 else f"    - {value.sample_rows}")
        except Exception as e:
            print(f"  ❌ 查询表 '{table_name}' 时出错: {e}")

    print("\n" + "=" * 80)
    print("查询完成")
    print("=" * 80)


if __name__ == "__main__":

    query_metadata(
        namespace='test',
        database_name='',
        table_name=''
    )
