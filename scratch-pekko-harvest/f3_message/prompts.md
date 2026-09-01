# F3 — actor Message enum + handler arm

1. Handle `Ping` by returning `Pong`.
2. Add `Inc` message that bumps an internal counter.
3. Non-exhaustive-safe match: unknown messages return `Ignored`.
4. Rename-resistant paraphrase: "when the actor receives Ping, reply Pong".
5. (Korean) Ping 메시지를 받으면 Pong을 반환하는 핸들러를 작성해.
