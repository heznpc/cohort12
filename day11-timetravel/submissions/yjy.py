import uuid
import json
from dotenv import load_dotenv
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


# =========================================================
# 상태 정의 (모든 필드 = 자동 persistence 대상)
# =========================================================

class AegisState(TypedDict, total=False):
    frame_id: str
    frame_meta: str

    # VLM
    vlm_status: str     # 정상/의심/이상
    vlm_class: str      # 절도/파손/실신/폭행/투기/none
    vlm_report: str     # 사실 묘사

    # LLM
    final_label: str
    decision: str
    final_report: str


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# =========================================================
# util : JSON 안전 파싱 (LLM 가끔 ```json 붙임 방지)
# =========================================================

def safe_json(text: str):
    text = text.strip().replace("```json", "").replace("```", "")
    return json.loads(text)


# =========================================================
# 1. VLM (감지 + 분류 + 묘사)
# =========================================================

def vlm_perception(state: AegisState):

    prompt = f"""
객체와 행동만 사실 그대로 묘사하고 판단하지 마.

반드시 JSON만 출력:
{{
 "status": "정상|의심|이상",
 "class": "절도|파손|실신|폭행|투기|none",
 "report": "사실 묘사 한 문장"
}}

장면: {state['frame_meta']}
"""

    res = model.invoke(prompt)
    data = safe_json(res.content)

    return {
        "vlm_status": data["status"],
        "vlm_class": data["class"],
        "vlm_report": data["report"]
    }


# =========================================================
# 2. LLM 검증 + 대응 + 보고서
# =========================================================

def llm_validation(state: AegisState):

    prompt = f"""
다음 VLM 결과를 검증하라.

status={state['vlm_status']}
class={state['vlm_class']}
report={state['vlm_report']}

1) 최종 이상 여부 판단
2) 대응 액션 결정
3) 육하원칙 보고서 작성

JSON만 출력:
{{
 "final_label": "정상|이상",
 "decision": "행동 한 줄",
 "report": "상황 보고서"
}}
"""

    res = model.invoke(prompt)
    data = safe_json(res.content)

    return {
        "final_label": data["final_label"],
        "decision": data["decision"],
        "final_report": data["report"]
    }


# =========================================================
# Graph 구성
# =========================================================

builder = StateGraph(AegisState)

builder.add_node("vlm_perception", vlm_perception)
builder.add_node("llm_validation", llm_validation)

builder.add_edge(START, "vlm_perception")
builder.add_edge("vlm_perception", "llm_validation")
builder.add_edge("llm_validation", END)


# =========================================================
# 실행 (⭐ checkpointer 반드시 with 사용)
# =========================================================

if __name__ == "__main__":

    config = {"configurable": {"thread_id": "demo"}}

    # ⭐⭐⭐ 여기 핵심 수정 ⭐⭐⭐
    with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:

        graph = builder.compile(checkpointer=checkpointer)

        result = graph.invoke(
            {
                "frame_id": str(uuid.uuid4())[:8],
                "frame_meta": "야간 공장, 남성 한 명이 바닥에 쓰러져 움직이지 않음"
            },
            config
        )

        print("\n===== 최종 결과 =====")
        print(json.dumps(result, indent=2, ensure_ascii=False))
