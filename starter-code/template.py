"""
Lab #3: Baseline Chatbot vs ReAct Agent
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.
"""

import json
import os
from tools import TOOL_DEFINITIONS, TOOL_MAP, get_flight_info, get_weather_forecast
from openai import OpenAI
from typing import List, Dict, Any

client = OpenAI()

SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng Vingroup.
Bạn chỉ sử dụng các công cụ sau:
{tools}

Quy trình trả lời bắt buộc:
Thought: <Suy nghĩ bước tiếp theo>
Action: {{"name": "<tên tool>", "args": {{<tham số>}}}}
Observation: <Kết quả từ tool>
... (Lặp lại cho tới khi có đủ dữ liệu)
Final Answer: <Câu trả lời hoàn chỉnh cho khách hàng>
"""

class ChatbotBaseline:
    """Baseline LLM Chatbot (Không sử dụng ReAct Loop hay Tools)"""
    def query(self, user_input: str) -> str:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Bạn là chatbot hỗ trợ khách hàng. "
                        "Hãy trả lời câu hỏi của người dùng. "
                        "Bạn không có quyền truy cập vào công cụ, "
                        "cơ sở dữ liệu hoặc dữ liệu thời gian thực."
                    ),
                },
                {
                    "role": "user",
                    "content": user_input,
                },
            ],
        )

        return response.choices[0].message.content

class ReActAgent:
    """ReAct Agent có sử dụng Thought-Action-Observation Loop"""

    def __init__(self, max_iterations: int = 5, api_key: str = None):
        self.max_iterations = max_iterations
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.trace: List[Dict[str, Any]] = []

    def run(self, user_input: str) -> str:
        # TODO 1: Khởi tạo mảng lưu lịch sử conversation / traces
        self.trace = []

        # TODO 2: Thiết lập vòng lặp while iteration < self.max_iterations
        iteration = 0
        messages = [{"role": "user", "content": user_input}]

        while iteration < self.max_iterations:
            iteration += 1

            # TODO 3: Phân tích Thought / Action từ Agent
            system_prompt = SYSTEM_PROMPT.format(
                tools=json.dumps(
                    TOOL_DEFINITIONS,
                    indent=2,
                    ensure_ascii=False
                )
            )

            messages_with_system = [
                {"role": "system", "content": system_prompt}
            ] + messages

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages_with_system,
                temperature=0
            )

            agent_response = response.choices[0].message.content.strip()

            trace_item = {
                "iteration": iteration
            }

            if "Thought:" in agent_response:
                thought = agent_response.split("Thought:", 1)[1]

                if "Action:" in thought:
                    thought = thought.split("Action:", 1)[0]

                elif "Final Answer:" in thought:
                    thought = thought.split("Final Answer:", 1)[0]

                trace_item["thought"] = thought.strip()

            # TODO 4: Thực thi Tool trong TOOL_MAP nếu có Action
            if "Action:" in agent_response:
                action_text = agent_response.split(
                    "Action:", 1
                )[1].strip()

                try:
                    json_start = action_text.find("{")
                    decoder = json.JSONDecoder()

                    action_json, _ = decoder.raw_decode(
                        action_text[json_start:]
                    )

                    tool_name = action_json.get("name")
                    tool_args = action_json.get("args", {})

                    if tool_name not in TOOL_MAP:
                        return f"Tool '{tool_name}' không tồn tại."

                    observation = TOOL_MAP[tool_name](**tool_args)

                    trace_item["action"] = action_json
                    trace_item["observation"] = observation

                    # TODO 5: Ghi lại Observation và lặp lại cho tới khi ra Final Answer
                    self.trace.append(trace_item)

                    messages.append({
                        "role": "assistant",
                        "content": agent_response
                    })

                    messages.append({
                        "role": "user",
                        "content": (
                            f"Observation: "
                            f"{json.dumps(observation, ensure_ascii=False)}"
                        )
                    })

                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    trace_item["error"] = str(e)
                    self.trace.append(trace_item)
                    return "Không thể phân tích Action của Agent."

            elif "Final Answer:" in agent_response:
                final_answer = agent_response.split(
                    "Final Answer:", 1
                )[1].strip()

                trace_item["final_answer"] = final_answer
                self.trace.append(trace_item)

                return final_answer

            else:
                self.trace.append(trace_item)
                return "Agent không trả về Action hoặc Final Answer."

        return "No answer found after maximum iterations"

def main():
    user_query = "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, rồi cho biết thời tiết SGN nên mặc gì?"
    
    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))
    
    print("\n=== RUNNING REACT AGENT ===")
    agent = ReActAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result)
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()