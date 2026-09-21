import copy

from .agent import Agent, ReflexAgent
from .interceptor import Interceptor
from .memory import MemoryEntry, SharedMemory
from .prompts import Phase


class Network:
    """
    1リクエスト分のAgent群の入れ物。

        net = Network(agents=[front, librarian])
        response = net["front"].respond(message="...")

    構築時にSharedMemoryを1つ作って全Agentへ配るため、インスタンスの寿命が
    セッションの寿命になる。Agent実体を使い回す場合はreset()を呼ぶ。

    担うのはAgent単体では決められない配線だけ。

      1. SharedMemoryを1つ作り、全Agentへ同じ実体を渡す
      2. sub_agent_names（文字列）をsub_agents（実体）へ解決する
      3. interceptorの既定値を配る
      4. Toolインスタンスを複製する（無効化が他のAgentへ波及しないように）
      5. 配線の誤りを構築時に落とす（存在しない委譲先、循環、初期toolの引数不整合）

    llmは配らない。接続先はAgentごとに違うのが前提のため。
    """

    def __init__(
        self,
        *,
        agents: list[Agent],
        interceptor: Interceptor | None = None,
        preload: dict[str, list[str]] | None = None,
    ):
        if not agents:
            raise ValueError("agentsを1つ以上指定してください")

        # 記憶は常に空で始まる。引き継ぎは構築後にshared_memoryへ代入する
        # （全Agentが同じ実体を参照するため、代入は全員へ反映される）。
        self.shared_memory = SharedMemory()
        self.interceptor = interceptor or Interceptor()
        self.agents: dict[str, Agent] = {}

        for agent in agents:
            if agent.name in self.agents:
                raise ValueError(f"Agent名が重複しています: {agent.name}")
            self.agents[agent.name] = agent

        self._wire()
        self._resolve_sub_agents()
        self._detect_cycles()
        self._preload(preload or {})

    # ---- 配線 ----
    def _wire(self) -> None:
        for agent in self.agents.values():
            agent.shared_memory = self.shared_memory

            # Toolを複製する。共有したままだと、あるAgentが立てたdisabledが
            # 他のAgentにも、次のリクエストにも残る（浅いコピーで足りる）。
            # 複製元は最初に渡されたTool。agent.toolsは下で置き換えるので、
            # 2回目の配線でそこから複製すると無効化済みのコピーが元になる。
            if agent._source_tools is None:
                agent._source_tools = agent.tools
            agent.tools = [copy.copy(t) for t in agent._source_tools]

            # Network共通のinterceptorを配る。個別に登録済みならそちらを尊重する。
            if not agent.interceptor.has_listeners:
                agent.interceptor = self.interceptor

            # llmだけ差し替えてmodelを書き忘れると、別プロバイダのモデル名を
            # 送ることになる。APIは「そんなモデルは無い」しか返さず原因が
            # 見えないため、構築時に確かめる。
            for phase, override in agent.phase_overrides.items():
                if override.llm is not None and override.model is None:
                    raise ValueError(
                        f"{agent.name}のphase_overrides[{phase.name}]がllmだけを"
                        f"上書きしています。接続を変えるならmodelも指定してください"
                        f"（モデル名はプロバイダごとに異なるため、"
                        f"既定の '{agent.model}' がそのまま使われると食い違います）。"
                    )

            # ReflexAgentが通らないphaseを上書きしていないか。
            # 記憶を作るphaseは持たない。ANSWERも、呼ぶ対象があればテキストを
            # 返した時点でループを抜けるため通らない。逆に対象を持たない
            # ReflexAgent（単発の判定・分類）ではANSWERが唯一通るphaseになる。
            if isinstance(agent, ReflexAgent):
                dead = {Phase.INITIAL_MEMORY, Phase.MEMORY_UPDATE}
                if agent.tools or agent.sub_agent_names or agent.sub_agents:
                    dead.add(Phase.ANSWER)
                for phase in agent.phase_overrides:
                    if phase not in dead:
                        continue
                    if phase is Phase.ANSWER:
                        reason = (
                            "ReflexAgentはテキストを返した時点でループを抜けるため、"
                            "利用者が読む回答文はFUNCTION_CALLフェーズが生成します"
                            "（output_schemaはsystem_instructionの指示文として伝わります）。"
                            "ANSWERへ入るのは、提示できる対象が実行時に全て無効化された場合だけです。"
                            "回答文のモデルを変えたいならAgent自身のmodelを指定してください。"
                        )
                    else:
                        reason = (
                            "ReflexAgentは記憶を作るphaseを持たないため、"
                            "このphaseの生成は一度も発生しません。"
                        )
                    raise ValueError(
                        f"{agent.name}のphase_overrides[{phase.name}]は効きません。{reason}"
                    )

            # 実行時に落とすとexecute()の例外処理に飲まれ、「なぜか初期toolが
            # 動かない」形になるため、構築時に確かめる。
            seen: set[str] = set()
            available = set((agent.input_schema or {}).get("properties", {}))
            for target in agent.initial_tools:
                # 同じものを2度並べても、同一の名前と引数で2回呼ぶだけになる
                # （2回目は重複として拒否される）。設定ミス以外に理由が無い。
                if target.name in seen:
                    raise ValueError(
                        f"{agent.name}のinitial_toolsに '{target.name}' が重複しています。"
                    )
                seen.add(target.name)

                # agentを置く場合、配線を受けていない実体だと自分だけの空の
                # 共有記憶で動く。sub_agentsと同じ理由で同じチェックをする。
                if isinstance(target, Agent) and self.agents.get(target.name) is not target:
                    raise ValueError(
                        f"{agent.name}のinitial_tools内の'{target.name}'がagentsに"
                        f"含まれていません。Networkの配線を受けないため、"
                        f"shared_memoryが共有されません。"
                    )

                # 初期ステップの引数はinput_schemaの同名プロパティから渡される。
                # 噛み合っていないと引数ゼロで呼ばれ、全てに既定値があると
                # 依頼と無関係な結果が事実としてmemoryへ書かれる。
                schema = target.to_declaration().get("parameters") or {}
                if missing := set(schema.get("required", [])) - available:
                    raise ValueError(
                        f"{agent.name}のinitial_tools '{target.name}' が必要とする引数"
                        f" {sorted(missing)} が、このAgentのinput_schemaに存在しません。"
                        f"input_schemaを合わせるか、別の対象を指定してください。"
                    )

    def _resolve_sub_agents(self) -> None:
        for agent in self.agents.values():
            # 利用側が実体で直接指定した分。agent.sub_agentsは下で解決済みの
            # 実体に置き換わるため、毎回そこから読むと2回目の構築で
            # 「名前と実体の併用」と誤判定される。
            if agent._source_sub_agents is None:
                agent._source_sub_agents = agent.sub_agents
            direct = agent._source_sub_agents

            resolved = []
            for name in agent.sub_agent_names:
                if name not in self.agents:
                    raise ValueError(f"{agent.name}が存在しないagentを参照しています: {name}")
                if name == agent.name:
                    raise ValueError(f"{agent.name}が自分自身を参照しています")
                resolved.append(self.agents[name])
            # 併用されると、どちらを採ってももう一方が黙って消える。
            if resolved and direct:
                raise ValueError(
                    f"{agent.name}がsub_agent_namesとsub_agentsを同時に指定しています。"
                    f"どちらか一方にしてください。"
                )
            # 配線を受けていない実体が委譲先に混ざると、shared_memoryが
            # 別インスタンスのまま実行される。
            for sub in direct:
                if self.agents.get(sub.name) is not sub:
                    raise ValueError(
                        f"{agent.name}のsub_agents内の'{sub.name}'がagentsに含まれていません。"
                        f"Networkの配線を受けないため、shared_memoryが共有されません。"
                    )
            agent.sub_agents = resolved or direct

    def _detect_cycles(self) -> None:
        """
        委譲の循環（A→B→A）を構築時に検出する。

        循環があると各Agentはmax_stepsで止まるが、委譲の深さが際限なく増える。
        実行時には「終わらない」という形でしか現れないため、ここで落とす。

        深さ優先で辿り、集合を2つ使う。

            visiting … 今辿っている経路上にいるAgent
            done     … 探索が終わったAgent（循環が無いと確定済み）

        訪問済みかどうかだけで判定すると、A→B→D / A→C→D のような合流を
        循環と誤判定する。
        """
        visiting, done = set(), set()

        def walk(agent: Agent, path: list[str]) -> None:
            # 既に調べ終わっているなら、辿り直す必要はない。
            if agent.name in done:
                return
            # 今辿っている経路上に再び現れたら、それが循環。
            if agent.name in visiting:
                loop = " → ".join([*path, agent.name])
                raise ValueError(f"委譲が循環しています: {loop}")

            visiting.add(agent.name)
            # 初期ステップの委譲先も委譲の辺として辿る。sub_agentsだけを見ると
            # A.initial_tools=[B] / B.sub_agents=[A] の循環が残り、実行時には
            # 再帰が止まらない形で現れる。
            for sub in (*agent.sub_agents, *agent.initial_tools):
                if not isinstance(sub, Agent):
                    continue
                # pathは、循環を見つけた時に経路をエラーへ出すために渡す。
                walk(sub, [*path, agent.name])
            # この枝を抜けるので、経路上からは外す。
            visiting.discard(agent.name)
            done.add(agent.name)

        # どのAgentからも辿れない孤立した部分が残らないよう、全員を起点にする。
        for agent in self.agents.values():
            walk(agent, [])

    def _preload(self, preload: dict[str, list[str]]) -> None:
        """
        セッション開始時点でSharedMemoryへ入れておく内容。

        privateは各Agentがセッション内で構築するため対象はsharedだけ。
        システムからの書き込みなので、システム所有プロパティの制限は受けない。
        """
        for field_name, texts in preload.items():
            bucket = getattr(self.shared_memory, field_name, None)
            # 型の不正ではなく設定の誤り（プロパティが無い、または配列でない）
            # なので、他の構築時検証と揃えてValueErrorにする。
            if not isinstance(bucket, list):
                raise ValueError(f"preloadできないプロパティです: {field_name}")  # noqa: TRY004
            for i, text in enumerate(texts, start=1):
                bucket.append(MemoryEntry(id=f"preload-{field_name}-{i}", text=text))

    # ---- 利用 ----
    def reset(self) -> None:
        """
        同じAgent実体で次のリクエストを始められる状態へ戻す。

        消える: 共有記憶 / 各Agentのprivate_memory / トークン累計 /
                セッション中の無効化 / 起動ごとの状態（依頼・履歴・周回数・添付）
        残る:   配線と構築時の検証結果 / Agentの設定

        Agentをモジュールレベルで定義して使い回す構成で、次のリクエストの
        手前で呼ぶ。自動では走らない——リクエストの区切りを知っているのは
        利用側だけで、構築時に消すと「構築後に引き継ぎを入れる」使い方と
        噛み合わないため。引き継ぎを入れるのはこのメソッドの後。

            network.reset()
            network.shared_memory.back_grounds = carried
        """
        # 空にするのではなく差し替える（項目が増えた時の消し忘れを防ぐ）。
        # reset()の前に取った参照は古い実体を指したままになる。
        self.shared_memory = SharedMemory()

        for agent in self.agents.values():
            agent.reset()

        # 新しい共有記憶と、_source_toolsから複製し直したtoolを配る
        # （設定としてのdisabledだけが残る）。
        self._wire()

    def get(self, name: str) -> Agent:
        if name not in self.agents:
            raise KeyError(f"存在しないagentです: {name}")
        return self.agents[name]

    def __getitem__(self, name: str) -> Agent:
        return self.get(name)

    def total_tokens(self) -> tuple[int, int]:
        """
        セッション全体の入力・出力トークン数。

        各Agentは自分の消費分だけを数えるため、単純な合計で二重計上が起きない。
        """
        return (
            sum(a.total_input_tokens for a in self.agents.values()),
            sum(a.total_output_tokens for a in self.agents.values()),
        )

    def token_usage(self) -> dict[str, tuple[int, int]]:
        """Agentごとのトークン消費の内訳。どのAgentが重いかを見るためのもの。"""
        return {
            name: (a.total_input_tokens, a.total_output_tokens) for name, a in self.agents.items()
        }


