# =========================================================
# Pokemon LangGraph Adventure (Fixed Pattern 2)
# =========================================================
import operator
import os
from typing import Annotated, List, TypedDict, Union

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, RemoveMessage

# LangGraph Core Imports
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# API Key 체크
if not os.getenv("GOOGLE_API_KEY"):
    print("⚠️ 경고: .env 파일에 GOOGLE_API_KEY가 없습니다.")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", # 모델명 확인
    temperature=0.7
)

# =========================================================
# 1. Battle Subgraph (전투 시스템)
# =========================================================

class BattleState(TypedDict):
    # MainState와 이름이 같아야 데이터가 자동으로 넘어옵니다.
    player_hp: int
    enemy_hp: int
    enemy_name: str
    battle_result: str
    # 로그는 계속 쌓여야 하므로 reducer 사용
    log: Annotated[List[str], operator.add]

def player_turn(state: BattleState):
    """플레이어 턴: 인터럽트 발생"""

    # 1. 사용자 입력 대기 (여기서 멈춤)
    skill = interrupt(
        f"[{state['enemy_name']} HP:{state['enemy_hp']}] "
        "어떤 기술을 쓸까? (전기/몸통박치기/도망)"
    )

    # 2. Resume 후 실행
    if skill == "도망":
        return {
            "battle_result": "escape",
            "log": ["🏃 플레이어가 도망쳤다!"]
        }

    dmg = 35 if skill == "전기" else 15
    new_hp = state["enemy_hp"] - dmg

    return {
        "enemy_hp": new_hp,
        "log": [f"⚡ 피카츄의 {skill} 공격! (데미지: {dmg})"]
    }

def enemy_turn(state: BattleState):
    """적 턴"""
    if state["enemy_hp"] <= 0:
        return {
            "battle_result": "win",
            "log": [f"🌟 {state['enemy_name']}이(가) 쓰러졌다!"]
        }

    dmg = 10
    new_hp = state["player_hp"] - dmg

    return {
        "player_hp": new_hp,
        "log": [f"💢 {state['enemy_name']}의 반격! (내 체력: {new_hp})"]
    }

def check_battle_end(state: BattleState):
    if state.get("battle_result") in ["win", "escape"]:
        return END
    if state["player_hp"] <= 0:
        return END
    return "player_turn"

# 서브그래프 구성
battle_builder = StateGraph(BattleState)
battle_builder.add_node("player_turn", player_turn)
battle_builder.add_node("enemy_turn", enemy_turn)

battle_builder.add_edge(START, "player_turn")
battle_builder.add_edge("player_turn", "enemy_turn")
battle_builder.add_conditional_edges(
    "enemy_turn",
    check_battle_end,
    {"player_turn": "player_turn", END: END}
)

# ★ 중요: 서브그래프 자체는 checkpointer 없이 컴파일 (부모가 관리함)
battle_subgraph = battle_builder.compile()


# =========================================================
# 2. Main Graph (모험 및 기억 관리)
# =========================================================

class MainState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    summary: str
    location: str

    # [수정 1] Pattern 2를 쓰려면 MainState에 배틀 변수가 있어야 함 (필수)
    player_hp: int
    enemy_hp: int
    enemy_name: str
    battle_result: str
    log: Annotated[List[str], operator.add]

def adventure_node(state: MainState):
    summary = state.get("summary", "모험을 막 시작했다.")

    prompt = f"""
    당신은 게임 마스터입니다. 현재 위치: {state.get('location', '태초마을')}
    지난 줄거리: {summary}
    사용자가 '풀숲'에 가면 "야생 포켓몬이 나타났다!"라고 하세요.
    """
    response = llm.invoke([SystemMessage(content=prompt)] + state["messages"])
    return {"messages": [response]}

def router(state: MainState):
    last_msg = state["messages"][-1].content
    if "야생 포켓몬" in last_msg or "승부" in last_msg:
        return "prepare_battle"
    return "memory_manager"

def prepare_battle(state: MainState):
    """
    [수정 2] 핵심 버그 수정 구간!
    배틀 시작 전, 이전 배틀의 변수들을 반드시 '초기화(Overwrite)' 해야 함.
    안 그러면 죽은 상태(HP 0)로 배틀이 시작됨.
    """
    print("\n⚠️ [시스템] 야생 포켓몬 출현! 배틀 데이터를 초기화합니다.")
    return {
        "player_hp": 100,        # 체력 리셋
        "enemy_hp": 60,          # 적 체력 리셋
        "enemy_name": "꼬렛",
        "battle_result": "ready",
        "log": []                # 로그 리셋 (이거 안 하면 이전 로그가 계속 남음)
    }

def handle_battle_result(state: MainState):
    result = state.get("battle_result")
    # 리스트로 된 로그를 문자열로 합침
    battle_logs = "\n".join(state.get("log", []))

    if result == "win":
        msg = "배틀 승리! 경험치를 획득했다."
    elif result == "escape":
        msg = "무사히 도망쳤다."
    else:
        msg = "눈앞이 깜깜해졌다... (패배)"

    final_msg = f"[배틀 기록]\n{battle_logs}\n\n[결과] {msg}"

    # [Tip] 배틀 끝났으니 메모리 정리를 위해 battle 변수들을 비워주는 것도 좋음 (선택)
    return {
        "messages": [SystemMessage(content=final_msg)]
    }

def memory_manager(state: MainState):
    msgs = state["messages"]
    if len(msgs) <= 6: return {}

    print("\n💾 [시스템] 기억 요약 중...")
    summary_res = llm.invoke([
        SystemMessage(content=f"요약해줘: {state.get('summary', '')}"),
        HumanMessage(content=str(msgs))
    ])

    # 시스템 메시지 제외하고 오래된 것 삭제
    del_msgs = [RemoveMessage(id=m.id) for m in msgs[:-2] if not isinstance(m, SystemMessage)]
    return {"summary": summary_res.content, "messages": del_msgs}

# 메인 그래프 조립
builder = StateGraph(MainState)

builder.add_node("adventure", adventure_node)
builder.add_node("prepare_battle", prepare_battle)

# ★ [수정 3] 서브그래프를 '노드'로 추가 (Pattern 2)
# 입력된 MainState가 그대로 battle_subgraph로 흘러들어가고,
# 배틀이 끝나면 변경된 값이 다시 MainState로 합쳐짐.
builder.add_node("battle_subgraph", battle_subgraph)

builder.add_node("battle_result", handle_battle_result)
builder.add_node("memory_manager", memory_manager)

builder.add_edge(START, "adventure")
builder.add_conditional_edges("adventure", router, {"prepare_battle": "prepare_battle", "memory_manager": "memory_manager"})

builder.add_edge("prepare_battle", "battle_subgraph")  # 준비 -> 배틀(서브그래프)
builder.add_edge("battle_subgraph", "battle_result")   # 배틀 끝 -> 결과 처리
builder.add_edge("battle_result", "memory_manager")
builder.add_edge("memory_manager", END)

app = builder.compile(checkpointer=InMemorySaver())


# =========================================================
# 3. Execution Loop (Interrupt Handling)
# =========================================================

def main():
    config = {"configurable": {"thread_id": "main_user_v1"}}
    print("🎮 [포켓몬] LangGraph Pattern 2 (Fixed)")
    print("💡 힌트: '풀숲' -> 배틀 -> 종료 후 다시 '풀숲' -> 새 배틀 가능\n")

    while True:
        try:
            # 1. 사용자 입력 받기
            user_input = input("👤 지우: ")
            if user_input.lower() in ["quit", "exit"]: break

            # 2. 그래프 실행
            # Command 없이 일반 실행
            result = app.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config
            )

            # 3. 결과 출력 (일반 대화)
            if result.get("messages"):
                print(f"🤖 도감: {result['messages'][-1].content}")

        except Exception:
            # 4. ★ Interrupt 발생 시 처리 로직 ★
            # invoke() 중에 interrupt가 걸리면 제어권이 여기로 옴 (예외 아님, 실행 종료됨)
            # LangGraph에서 interrupt는 실행을 '일시 정지'하고 리턴함.
            # 따라서 상태를 조회해서 인터럽트가 걸려있는지 확인해야 함.
            pass

        # 5. 실행 후 상태 확인 (Interrupt 체크)
        snapshot = app.get_state(config)

        # 다음 실행할 태스크가 있고, 그게 interrupt라면?
        if snapshot.next and snapshot.tasks[0].interrupts:
            # 인터럽트 값(질문) 추출
            question = snapshot.tasks[0].interrupts[0].value
            print(f"\n✋ [배틀 액션] {question}")

            # 배틀이 끝날 때까지 반복하는 내부 루프
            while snapshot.next and snapshot.tasks[0].interrupts:
                action = input("   > 선택: ")

                # Command를 사용해 멈춘 지점(resume)으로 값 전달
                result = app.invoke(Command(resume=action), config)

                # 실행 후 다시 상태 확인
                snapshot = app.get_state(config)

                # 만약 인터럽트가 또 있으면(다음 턴) 루프 반복, 없으면(배틀 종료) 탈출
                if snapshot.next and snapshot.tasks[0].interrupts:
                    question = snapshot.tasks[0].interrupts[0].value
                    print(f"\n✋ [배틀 액션] {question}")
                else:
                    # 배틀 종료 후 결과 메시지 출력
                    if "messages" in result and result["messages"]:
                        print(f"🤖 도감: {result['messages'][-1].content}")

if __name__ == "__main__":
    main()