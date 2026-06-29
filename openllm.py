#!/usr/bin/env python3
"""OpenLLM CLI — 交互式对话入口"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from openllm.core.engine import OpenLLMEngine, AgentConfig

def main():
    import argparse
    parser = argparse.ArgumentParser(description='OpenLLM Agent')
    parser.add_argument('--model', default='qwen3.5:9b', help='Ollama 模型名')
    parser.add_argument('--provider', default='ollama', help='Provider (ollama/deepseek)')
    parser.add_argument('--name', default='OpenLLM', help='Agent 名字')
    args = parser.parse_args()

    config = AgentConfig(
        name=args.name,
        provider=args.provider,
        model=args.model,
    )
    engine = OpenLLMEngine(config)

    # 唤醒
    wake_msg = engine.wake()
    print(wake_msg)
    print()

    # 交互循环
    print("输入消息开始对话（输入 /quit 退出，/status 查看状态）")
    print("-" * 50)

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input == '/quit':
            engine.sleep()
            print("记忆已保存。再见！")
            break
        if user_input == '/status':
            import json
            print(json.dumps(engine.status(), indent=2, ensure_ascii=False))
            continue
        if user_input == '/tools':
            for t in engine.tools.list_tools():
                print(f"  - {t['name']}: {t['description']}")
            continue

        response = engine.chat(user_input, stream=True)
        print()

if __name__ == '__main__':
    main()
