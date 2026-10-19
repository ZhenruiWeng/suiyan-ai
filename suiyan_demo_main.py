# -*- coding: utf-8 -*-
"""SuiYan AI - end-to-end demo.

Flow: mock SkinProfile JSON -> ingestion check -> retrieval -> plan generation
      -> anti-hallucination validation -> pretty JSON output.

Run from the RAGProject root:  py -3.12 suiyan_demo.py
Requires DASHSCOPE_API_KEY environment variable.
"""
import json
import os
import sys

from suiyan_chain import build_plan_chain, plan_to_json
from suiyan_product_store import SuiyanKnowledgeStore, SuiyanProductStore
from suiyan_schemas import SkinProfile

# ---------------------------------------------------------------------------
# Mock skin profiles (in production these come from the vision module)
# ---------------------------------------------------------------------------
MOCK_PROFILES = [
    {
        "skin_type": "干性",
        "skin_tone": "自然偏白（黄二白）",
        "undertone": "暖调",
        "concerns": ["法令纹", "眼周细纹", "两颊干燥", "暗沉"],
        "occasion": "日常通勤",
    },
    {
        "skin_type": "敏感肌",
        "skin_tone": "冷白皮",
        "undertone": "冷调",
        "concerns": ["泛红", "薄皮", "唇纹深"],
        "occasion": "重要活动",
    },
    {
        "skin_type": "混合性",
        "skin_tone": "自然健康肤色（黄三白）",
        "undertone": "暖调",
        "concerns": ["T区出油", "斑点", "毛孔"],
        "occasion": "宴会聚会",
    },
]


def main():
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("ERROR: DASHSCOPE_API_KEY not set.")
        sys.exit(1)

    # 1) Ingest (idempotent: skips if already built)
    product_store = SuiyanProductStore()
    n_products = product_store.ingest()
    knowledge_store = SuiyanKnowledgeStore()
    n_knowledge = knowledge_store.ingest()
    print(f"[setup] products={n_products}, knowledge_chunks={n_knowledge}")

    # 2) Build chain once
    run_plan, _ = build_plan_chain(product_store, knowledge_store)

    # 3) Run on requested (or first) mock profile
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    profile_dict = MOCK_PROFILES[idx % len(MOCK_PROFILES)]
    profile = SkinProfile(**profile_dict)

    print("\n" + "=" * 60)
    print("INPUT SkinProfile:")
    print(json.dumps(profile_dict, ensure_ascii=False, indent=2))
    print("=" * 60)

    plan = run_plan(profile)

    print("\nOUTPUT MakeupPlan:")
    print(plan_to_json(plan))
    print("=" * 60)
    print(f"products kept: {len(plan.products)}, steps: {len(plan.steps)}")


if __name__ == "__main__":
    main()
