# -*- coding: utf-8 -*-
"""SuiYan AI - makeup plan generation chain.

Pipeline:
  SkinProfile(JSON) -> product retrieval (filtered) + knowledge retrieval
    -> prompt(qwen3-max) -> JsonOutputParser(MakeupPlan)
    -> anti-hallucination post-validation (products must come from candidates;
       shades must exist in the product's shade list)

The chain intentionally does NOT use ChatTongyi.with_structured_output
(unreliable with DashScope); JsonOutputParser + prompt discipline is used
instead, followed by deterministic python-side validation.
"""
import json
import os
from typing import Dict, List

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatTongyi

from suiyan_product_store import SuiyanKnowledgeStore, SuiyanProductStore
from suiyan_schemas import MakeupPlan, SkinProfile, profile_to_text

# Model is configurable: some DashScope accounts don't have qwen3-max enabled.
CHAT_MODEL = os.environ.get("SUIYAN_CHAT_MODEL", "qwen-max")

# Shade-selection rules baked into the prompt (from collected shade data)
SHADE_RULES = """色号选择规则（必须严格遵守）：
1. 黄皮/暖调用户只能选W系或中性N系色号，禁止选C系（粉调）。
2. 冷白皮/冷调用户优先C系；中性调可从N系或W1W2尝试。
3. 肤色深浅对号入座：黄一白→W10/1W；黄二白→W20/2W；黄三白→W30/3W。
4. 原则"宁深勿白"：色号偏白会放大法令纹、造成脸脖色差。
5. 口红色号按底调匹配：暖调选暖红/豆沙/枫叶/土棕（619、294、130、114）；冷调选玫瑰/梅子（RC301、114）；重要场合可选气场红（666、294厚涂）。
6. 熟龄肌口红优先缎光/釉光滋润质地，纯哑光需标注"先润唇打底"。
7. 护肤品无色号，shade字段留null。"""

SYSTEM_PROMPT = """你是欧莱雅集团专业彩妆顾问"岁颜AI"，专为45岁以上熟龄女性设计妆容方案。

核心理念：不去皱纹、保留原生面部特征，仅优化气色与光泽。绝不推荐遮盖式厚妆。

你将收到：
【用户肤质画像】
【候选产品清单】(含色号)——只能从这个清单里选品！
【上妆知识参考】

任务：
1. 从候选清单中选出4-7个产品（底妆1个+护肤打底1-2个+口红/唇釉1-2个+其他1-2个），并逐项输出category字段。
2. 严禁品类错配：口红推荐只能选口红/唇釉类产品，绝不能套用粉底等底妆产品的名字；推荐理由必须与该产品真实类别、功效一致。如果候选中没有合适品类，宁缺毋滥。
3. 每个产品给出选定色号（按色号选择规则）、推荐理由、使用要点。
4. 生成大字版化妆步骤（7-10步，每步简短有力，针对熟龄肌：薄涂按压、局部定妆、保湿优先）。
5. 给出妆容定位和原生感说明（说明保留了什么原生特征、优化了什么气色）。

{format_instructions}

注意：products[].name必须与候选清单中的产品名称完全一致；shade必须是该产品色号列表中的色号ID或null。只输出JSON，不要输出其他任何文字。"""


def _candidates_text(candidates: List[Dict]) -> str:
    """Format retrieved candidate products (with full shade lists) for the prompt."""
    blocks = []
    for i, p in enumerate(candidates, 1):
        shades = "、".join(
            f"{s['shade_id']}({s.get('desc', '')})" for s in p.get("shades", [])
        ) or "无色号"
        blocks.append(
            f"{i}. {p['name']}\n"
            f"   类别:{p['category']} 系列:{p['series']} 价格:¥{p['price']} 质地:{p.get('texture','')}\n"
            f"   适合肤质:{'/'.join(p.get('suitable_skin', []))} 适配肤色:{'/'.join(p.get('tone_match', []))}\n"
            f"   色号: {shades}\n"
            f"   功效: {p.get('benefits', '')}"
        )
    return "\n".join(blocks)


def build_plan_chain(product_store: SuiyanProductStore,
                     knowledge_store: SuiyanKnowledgeStore,
                     model_name: str = CHAT_MODEL):
    """Build the full plan-generation chain. Returns a callable chain and
    the product store (needed for post-validation)."""
    parser = JsonOutputParser(pydantic_object=MakeupPlan)
    llm = ChatTongyi(model=model_name, temperature=0.3, top_p=0.8)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT + "\n\n" + SHADE_RULES),
        ("user", """【用户肤质画像】
{profile}

【候选产品清单】
{candidates}

【上妆知识参考】
{knowledge}

规则补充：
- 若画像中含【用户语音需求】，优先满足其场合/预算/偏好要求，并允许覆盖默认场合。
- 若语音需求含预算，请结合候选清单价格选品并在推荐理由中说明价格。
- 若语音需求与量化结果冲突（如用户自述冷白皮但ITA°判定暖黄二白），以量化结果为准并在reason中温和解释。

请生成妆容方案JSON。"""),
    ]).partial(format_instructions=parser.get_format_instructions())

    def retrieve_inputs(profile: SkinProfile) -> Dict[str, str]:
        user_req = (profile.user_request or "").strip()
        # voice request enriches the retrieval query (occasion/budget/prefs)
        retrieve_query = f"{profile_to_text(profile)} {user_req} 卡纹 提气色 上妆步骤"
        candidates = product_store.search_text(retrieve_query, profile, k=8)
        knowledge_docs = knowledge_store.search(retrieve_query, k=4)
        knowledge_text = "\n".join(d.page_content for d in knowledge_docs)
        req_block = f"\n【用户语音需求】（优先级高于默认场合）\n{user_req}\n" if user_req else ""
        return {
            "profile": profile_to_text(profile) +
                       (f"（详情：肤质{profile.skin_type}，肤色{profile.skin_tone}，"
                        f"底调{profile.undertone}，关注{ '、'.join(profile.concerns) or '无'}，"
                        f"场合{profile.occasion}）") + req_block,
            "candidates": _candidates_text(candidates),
            "knowledge": knowledge_text[:3000],
        }, candidates

    chain = prompt | llm | parser

    def run(profile: SkinProfile) -> MakeupPlan:
        inputs, candidates = retrieve_inputs(profile)
        raw = chain.invoke(inputs)
        plan = validate_plan(raw, candidates, undertone=profile.undertone)
        return plan

    return run, product_store


# ---------------------------------------------------------------------------
# Anti-hallucination post-validation
# ---------------------------------------------------------------------------
def _default_lip_shade(product: Dict, undertone: str) -> str:
    """Pick a lip shade id matching the undertone; fallback to first shade."""
    by_pref = {
        "暖调": ["619", "130", "294", "916"],
        "冷调": ["RC301", "114", "666"],
        "中性调": ["RC301", "114", "619"],
    }.get(undertone, [])
    shade_ids = [s["shade_id"] for s in product.get("shades", [])]
    for sid in by_pref:
        if sid in shade_ids:
            return sid
    return shade_ids[0] if shade_ids else None


def validate_plan(raw: Dict, candidates: List[Dict], undertone: str = "中性调") -> MakeupPlan:
    """Drop recommended products that are NOT in the retrieved candidates;
    null-out shades that don't exist in the product's shade list."""
    valid = {p["name"]: p for p in candidates}
    # also allow nickname matching
    nick_map = {p.get("nickname", ""): p for p in candidates if p.get("nickname")}
    lookup = {**valid, **nick_map}

    cleaned_products = []
    dropped = []
    for prod in raw.get("products", []):
        name = (prod.get("name") or "").strip()
        p = lookup.get(name)
        if p is None:
            # fuzzy: check substring containment both ways
            for cand_name, cand in lookup.items():
                if cand_name and (cand_name in name or name in cand_name):
                    p = cand
                    name = cand_name
                    break
        if p is None:
            dropped.append(prod.get("name"))
            continue
        # category consistency check: catches LLM using a foundation's name
        # for a lipstick recommendation, etc.
        if prod.get("category") and prod["category"] != p.get("category"):
            dropped.append(f"{prod.get('name')}(category mismatch: "
                           f"claimed {prod['category']}, actual {p.get('category')})")
            continue
        shade = prod.get("shade")
        if shade and p.get("shades"):
            if not any(s["shade_id"] == shade for s in p["shades"]):
                shade = None  # invalid shade -> null, keep the product
        cleaned_products.append({
            "name": p["name"],
            "category": p.get("category", ""),
            "shade": shade,
            "reason": prod.get("reason", ""),
            "usage_hint": prod.get("usage_hint", ""),
        })

    # Category backfill: a makeup plan MUST contain one lip product and one
    # base product; if the LLM skipped them, auto-append from candidates.
    def _has_cat(cat_set):
        return any(p["category"] in cat_set for p in cleaned_products)

    if not _has_cat({"口红", "唇釉"}):
        for c in candidates:
            if c.get("category") in {"口红", "唇釉"}:
                cleaned_products.append({
                    "name": c["name"],
                    "category": c["category"],
                    "shade": _default_lip_shade(c, undertone),
                    "reason": f"系统按{undertone}底调补充的显气色唇妆，熟龄肌提气色关键单品",
                    "usage_hint": "先润唇打底再涂，薄涂日常、厚涂提升气场",
                })
                print("[category-backfill] lip product appended:", c["name"])
                break
    if not _has_cat({"粉底", "气垫"}):
        for c in candidates:
            if c.get("category") in {"粉底", "气垫"}:
                cleaned_products.append({
                    "name": c["name"], "category": c["category"],
                    "shade": None, "reason": "系统补充的底妆单品，养肤轻质优先",
                    "usage_hint": "湿海绵薄涂按压",
                })
                print("[category-backfill] base product appended:", c["name"])
                break

    plan = MakeupPlan(
        positioning=raw.get("positioning", "熟龄气色妆"),
        products=cleaned_products,
        steps=raw.get("steps", []),
        native_note=raw.get("native_note", ""),
    )
    if dropped:
        print(f"[anti-hallucination] dropped non-catalog products: {dropped}")
    return plan


def plan_to_json(plan: MakeupPlan) -> str:
    return json.dumps(plan.model_dump(), ensure_ascii=False, indent=2)
