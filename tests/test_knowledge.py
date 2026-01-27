import argparse
import os
from pathlib import Path

# 导入核心类
from datus.storage.ext_knowledge.store import ExtKnowledgeStore
from datus.storage.ext_knowledge.ext_knowledge_init import init_ext_knowledge
from datus.storage.subject_tree.store import SubjectTreeStore
from datus.storage.metric.store import MetricStorage
from datus.storage.reference_sql.store import ReferenceSqlStorage
from datus.storage.embedding_models import EMBEDDING_MODELS

# 1. 初始化存储
db_path = "./data/datus_storage"
embedding_model = EMBEDDING_MODELS["ext_knowledge"]

# 2. 创建存储实例
ext_knowledge_store = ExtKnowledgeStore(db_path, embedding_model)
subject_tree_store = SubjectTreeStore(db_path)
metric_store = MetricStorage(db_path, embedding_model)
sql_store = ReferenceSqlStorage(db_path, embedding_model)

# 3. 准备参数
args = argparse.Namespace(
    ext_knowledge="/path/to/knowledge.csv",
    subject_tree="Sales/Reporting,Dashboard/Sales,Finance/Revenue"
)

# 4. 执行导入
init_ext_knowledge(
    storage=ext_knowledge_store,
    args=args,
    build_mode="incremental",  # 或 "overwrite"
    pool_size=4
)

print("导入完成！")
