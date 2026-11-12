"""Talk to your model.

    python -m modeltune.chat CONFIG --prompt "Barack Obama visited Berlin."   # one shot
    python -m modeltune.chat CONFIG                                            # interactive
    python -m modeltune.chat CONFIG --base                                     # the original model
"""

from __future__ import annotations

from .config import cli
from .evaluate import resolve_adapter
from .hardware import detect
from .inference import generate, load_for_inference


def main() -> None:
    def flags(p):
        p.add_argument("--base", action="store_true")
        p.add_argument("--adapter", default=None)
        p.add_argument("--prompt", default=None)
        p.add_argument("--system", default=None, help="system prompt (defaults to data.system_prompt)")
        p.add_argument("--max_new_tokens", type=int, default=512)
        p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")

    args, cfg = cli("Chat with a (fine-tuned) model.", flags)
    hw = detect()
    model, tok = load_for_inference(cfg, hw, resolve_adapter(cfg, args.base, args.adapter))
    system = args.system or cfg["data"]["system_prompt"]
    sampling = {"do_sample": True, "temperature": args.temperature} if args.temperature > 0 else {}

    history = [{"role": "system", "content": system}] if system else []

    def reply(text: str) -> str:
        history.append({"role": "user", "content": text})
        answer = generate(model, tok, [history], args.max_new_tokens, 1, cfg["eval"]["chat_template_kwargs"], **sampling)[0]
        history.append({"role": "assistant", "content": answer})
        return answer

    if args.prompt:
        print(reply(args.prompt))
        return
    print("Type a message (empty line or Ctrl-D to quit).")
    while True:
        try:
            text = input("\nyou> ").strip()
        except EOFError:
            break
        if not text:
            break
        print(f"model> {reply(text)}")


if __name__ == "__main__":
    main()
