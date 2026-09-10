import copy


from .agent import Agent
from .interceptor import Interceptor
from .memory import MemoryEntry, SharedMemory




class Network:
    """
    1つのセッション（＝1リクエスト）を構成するAgent群の入れ物。


    使い方:
        net = Network(agents=[front, librarian])
        response = net["front"].respond(message="...")


    【1リクエストにつき1インスタンス作る】
    Networkを作ると新しいSharedMemoryが1つ生成され、全Agentへ注入される。
    つまりインスタンスの寿命がセッションの寿命になる。使い回すと、
    前のリクエストの記憶やツール実行履歴が残ったまま次の依頼が処理される。


    Pythonのプロセスは起動したまま複数のリクエストを捌くため、
    モジュールの先頭で定義したAgentやToolを使い回すと状態が持ち越される。
    リクエストごとにNetworkを作り直すことで、そこを断ち切る。


    【Networkが担うこと】
    Agentの生成自体は利用側が行う。Networkは「Agent単体では決められない配線」
    だけを引き受ける。


      1. SharedMemoryを1つ作り、全Agentへ同じ実体を渡す
         → エージェント同士が報告し合うのではなく、同じ場所を読み書きする形になる
      2. sub_agent_names（文字列）をsub_agents（実体）へ解決する
         → 利用側は名前で書けばよく、相互参照のためにインスタンスの生成順を
           気にする必要がなくなる
      3. interceptorの既定値を配る
      4. Toolインスタンスを複製する
         → あるAgentがツールを無効化しても、同じToolを持つ他のAgentへ
           波及しないようにする
      5. 配線の誤りを構築時に検出して落とす
         → 存在しない委譲先、循環参照、初期ツールの引数不整合など。
           推論を1回走らせてから気付くべきものではないため、ここで落とす


    【llmを配らない理由】
    接続先はAgentごとに違うのが前提（あるAgentはGemini、別のAgentはClaude）
    なので、Network側に既定値を置かない。置くと「このAgentの接続先はどこで
    決まったのか」が追いにくくなる。
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


        # 記憶は常に空で始まる。前のターンから何を引き継ぐかは利用側の判断であり、
        # 構築後に shared_memory のプロパティへ代入すればよい。
        # 全Agentが同じインスタンスを参照するため、構築後の代入も全員に反映される。
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


    # ==========================================
    # 配線
    # ==========================================
    def _wire(self) -> None:
        for agent in self.agents.values():
            agent.shared_memory = self.shared_memory


            # 同じToolインスタンスを複数Agentで共有していると、あるAgentが
            # disabled=Trueにした時に他のAgentからもツールが消える。
            # さらにPythonはプロセスが生き続けるため、モジュールレベルで定義した
            # Toolを使い回すと次のリクエストにまでdisabledが残る。
            # 浅いコピーで十分（funcは共有されて問題なく、boolやstrだけが独立する）。
            #
            # 複製元は「最初に渡されたTool」でなければならない。agent.toolsを
            # 破壊的に置き換えているため、2回目以降の配線でそのまま複製すると、
            # 前回のセッション中にdisabled=Trueにされたコピーが複製元になり、
            # 結局リクエストをまたいで無効化が残る（防ごうとしたことが起きる）。
            if agent._source_tools is None:
                agent._source_tools = agent.tools
            agent.tools = [copy.copy(t) for t in agent._source_tools]


            # Network共通のinterceptorを配る。個別に登録済みならそちらを尊重する。
            if not agent.interceptor.has_listeners:
                agent.interceptor = self.interceptor


            # phase_overridesでllm（接続先）だけを差し替え、modelを書き忘れていないか。
            #
            # 接続とモデル名を別項目にした結果、片方だけ上書きすると
            # 「ClaudeへGeminiのモデル名を渡す」という食い違いが起きる。
            # 実行時にはそのphaseへ到達するまで分からず、しかもAPIからは
            # 「そんなモデルは無い」という汎用エラーしか返らないため、
            # 原因がphase_overridesの書き忘れだと気付きにくい。
            #
            # モデル名はプロバイダごとの語彙なので、接続を変えるなら
            # モデル名も一緒に変わるのが正しい。構築時に確かめる。
            for phase, override in agent.phase_overrides.items():
                if override.llm is not None and override.model is None:
                    raise ValueError(
                        f"{agent.name}のphase_overrides[{phase.name}]がllmだけを"
                        f"上書きしています。接続を変えるならmodelも指定してください"
                        f"（モデル名はプロバイダごとに異なるため、"
                        f"既定の '{agent.model}' がそのまま使われると食い違います）。"
                    )


            # initial_tool_nameの解決可否は構築時に確かめる。
            # 実行時に落とすと、execute()の例外処理に飲まれて
            # 「なぜか初期toolが動かない」状態になり、原因が見えなくなる。
            if agent.initial_tool_name:
                initial_tool = agent.get_tool(agent.initial_tool_name)
                if initial_tool is None:
                    raise ValueError(
                        f"{agent.name}のinitial_tool_name '{agent.initial_tool_name}' が"
                        f"toolsの中に見つかりません。"
                    )
                # 初期toolの引数は、このAgentが受け取る依頼（input_schema）から
                # 名前が一致するものだけを抜き出して渡す。噛み合っていないと
                # 引数ゼロで呼ばれ、必須引数があれば原因不明の汎用エラー、
                # 全てに既定値があれば「依頼と無関係な既定値で成功実行され、
                # その結果が事実としてmemoryへ書かれる」という汚染になる。
                # 実行時には気付けないので構築時に確かめる。
                required = set(initial_tool.get_json_schema().get("required", []))
                available = set((agent.input_schema or {}).get("properties", {}))
                missing = required - available
                if missing:
                    raise ValueError(
                        f"{agent.name}のinitial_tool '{agent.initial_tool_name}' が必要とする引数"
                        f" {sorted(missing)} が、このAgentのinput_schemaに存在しません。"
                        f"input_schemaを合わせるか、別のtoolを指定してください。"
                    )


    def _resolve_sub_agents(self) -> None:
        for agent in self.agents.values():
            resolved = []
            for name in agent.sub_agent_names:
                if name not in self.agents:
                    raise ValueError(f"{agent.name}が存在しないagentを参照しています: {name}")
                if name == agent.name:
                    raise ValueError(f"{agent.name}が自分自身を参照しています")
                resolved.append(self.agents[name])
            # 名前指定と実体の直接指定を併用されると、どちらを採用しても
            # もう一方が黙って消える。配線ミスとして構築時に落とす。
            if resolved and agent.sub_agents:
                raise ValueError(
                    f"{agent.name}がsub_agent_namesとsub_agentsを同時に指定しています。"
                    f"どちらか一方にしてください。"
                )
            # 直接sub_agentsを渡していた場合（単体実行など）はそれを残すが、
            # Networkの配線を受けていない実体が委譲先に混ざると、
            # shared_memoryが別インスタンスのまま実行されてしまう。
            for sub in agent.sub_agents:
                if self.agents.get(sub.name) is not sub:
                    raise ValueError(
                        f"{agent.name}のsub_agents内の'{sub.name}'がagentsに含まれていません。"
                        f"Networkの配線を受けないため、shared_memoryが共有されません。"
                    )
            agent.sub_agents = resolved or agent.sub_agents


    def _detect_cycles(self) -> None:
        """
        委譲の循環（A→B→A のような参照）を構築時に検出する。


        循環があると、各Agentはmax_stepsで止まるものの、
        委譲の深さが際限なく増える。AがBを呼び、BがAを呼び、
        そのAがまたBを呼ぶ、という形で階層が積み上がっていく。
        実行時には「なぜか終わらない」という形でしか現れないので、
        構築時に落とす。


        委譲関係を有向グラフとみなし、深さ優先で辿って循環を探す。
        2つの集合を使う点が要点。


            visiting … 今まさに辿っている経路上にいるAgent
            done     … 探索が完全に終わったAgent（循環が無いと確定済み）


        なぜ集合が2つ必要か:
        1つ（訪問済みかどうかだけ）で済ませると、次のような合流を
        誤って循環と判定してしまう。


            A → B → D
            A → C → D      （Dへ2つの経路から到達するが、循環はしていない）


        Dは2回訪問されるが、これは循環ではない。
        「経路上にいる（visiting）」と「もう調べ終わった（done）」を
        区別することで、合流と循環を見分けられる。
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
            for sub in agent.sub_agents:
                # pathを渡していくのは、循環を見つけた時に
                # 「どう辿って戻ってきたか」をエラーへ出すため。
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


        privateは各Agentがセッション内で構築するものなので、対象はsharedだけ。
        システムからの書き込みなので、システム所有プロパティの制限は受けない。
        """
        for field_name, texts in preload.items():
            bucket = getattr(self.shared_memory, field_name, None)
            # isinstanceで見ているが、これは型の不正ではなく設定の誤り
            # （そんな名前のプロパティが無い、または配列ではないので
            # 追記できない）。他の構築時検証と同じくValueErrorで揃える。
            if not isinstance(bucket, list):
                raise ValueError(f"preloadできないプロパティです: {field_name}")  # noqa: TRY004
            for i, text in enumerate(texts, start=1):
                bucket.append(MemoryEntry(id=f"preload-{field_name}-{i}", text=text))


    # ==========================================
    # 利用
    # ==========================================
    def get(self, name: str) -> Agent:
        if name not in self.agents:
            raise KeyError(f"存在しないagentです: {name}")
        return self.agents[name]


    def __getitem__(self, name: str) -> Agent:
        return self.get(name)


    def total_tokens(self) -> tuple[int, int]:
        """
        セッション全体の入力・出力トークン数。


        各Agentは自分が消費した分だけを数え、委譲先の分を含めない。
        そのため単純な合計で二重計上が起きない。
        """
        return (
            sum(a.total_input_tokens for a in self.agents.values()),
            sum(a.total_output_tokens for a in self.agents.values()),
        )


    def token_usage(self) -> dict[str, tuple[int, int]]:
        """
        Agentごとのトークン消費の内訳。


        どのAgentがコストを使っているかを特定するためのもの。
        委譲先の分が呼び出し元へ合算されないのは、この内訳を成立させるため。
        """
        return {
            name: (a.total_input_tokens, a.total_output_tokens) for name, a in self.agents.items()
        }





