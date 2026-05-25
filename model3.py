import mesa
from agent23 import SocialAgent
from network import generate_network
from datetime import datetime, timedelta
from mesa.datacollection import DataCollector
import dill as pickle
from tqdm import tqdm
from utils3 import *
from prompt2 import *
from collections import defaultdict, Counter
import re


def load_network_structure(file_path):
    with open(file_path, "r") as file:
        network_data = json.load(file)

    G = nx.Graph()

    G.add_nodes_from(network_data["nodes"])

    G.add_edges_from(network_data["edges"])

    return G


class World(mesa.Model):
    def __init__(self, num_agents, leaders, gpt_model, network_type="scale_free", load_network=False,
                 belief_keywords_file=None, exp_name="default_exp",
                 mitigation_perspectives_file=None,
                 exp_dir="./experiments", mitigation_step=1000, with_long_memory=True,
                 mitigation_perspectives_only=False,
                 topic="euthanasia", temp=0.5, recommendation="random", seed=50, **kwargs):
        super().__init__()
        self.belief_keywords = self.load_belief_keywords_file(belief_keywords_file)
        self.num_agents = num_agents
        self.network_type = network_type
        self.step_count = kwargs.get('step_count', 2)
        self.name = exp_name
        self.run_dir = os.path.join(exp_dir, self.name)
        self.temp = temp
        self.recommendation = recommendation
        self.backgrounds = {}
        self.leaders = leaders
        self.leaders_pos = {}
        self.mitigation_step = mitigation_step
        self.with_long_memory = with_long_memory
        self.mitigation_perspectives_only = mitigation_perspectives_only

        self.gpt_model = gpt_model

        os.makedirs(self.run_dir, exist_ok=True)
        self.opinion_templates = self.load_opinions()
        self.topic = self.opinion_templates[topic]
        self.topic1 = topic

        # 引入话题
        self.topics = list(self.opinion_templates.keys())

        self.current_date = datetime(2024, 1, 1)
        self.max_interactions = kwargs.get('max_interactions', 5)

        # self.recent_interact_topic = defaultdict(list)  # 存储每个 agent 当前轮选择的话题列表
        # self.topic_heat = Counter()  # 本轮话题热度统计
        

        if mitigation_perspectives_file is not None:
            with open(mitigation_perspectives_file, 'r') as json_file:
                self.mitigation_perspectives = json.load(json_file)["perspectives"]
                print(self.mitigation_perspectives)
        else:
            self.mitigation_perspectives = mitigation_perspectives_file

        network_file = f"./data/{network_type}_network_num_agents_{num_agents}_seed_{seed}.json"

        # 初始化三话题的信念值
        belief_file = f"./data1/numeric_sim_opnions_and_stubbornness_num_agents_{num_agents}_2topics.json"# 要改
        # 指定文件路径

        if self.gpt_model == "gpt-4o-mini-2024-07-18":
            gpt_model_name = "gpt-4o-mini"
        else:
            gpt_model_name = "qwen"

        # backgrounds_file = f"./data/agents_backgrounds_num_agents_{num_agents}_{topic}_{gpt_model_name}.json"
        backgrounds_file = f"./data1/agents_backgrounds_num_agents_{num_agents}_{topic}_{gpt_model_name}_2topics.json"

        if load_network and network_file:
            self.G = load_network_structure(network_file)
        else:
            self.G = generate_network(network_type, num_agents, **kwargs)
        self.grid = mesa.space.NetworkGrid(self.G)    # 将智能体放在图结构上
        self.schedule = mesa.time.RandomActivation(self)    # 使智能体按照随机顺序去行动


        # 获取话题列表
        self.topics = list(self.opinion_templates.keys())

        # 判断 topics 对应的 value 是否完全相同
        topic_values = [self.opinion_templates[t] for t in self.topics]
        if len(set(topic_values)) == 1:
            # 所有 value 相同 → 调用 generate_balanced_beliefs_dict1（生成相同信念）
            beliefs = self.generate_balanced_beliefs_dict1(self.num_agents)
        else:
            # 有不同 value → 调用原来的 generate_balanced_beliefs_dict（独立生成信念）
            beliefs = self.generate_balanced_beliefs_dict(self.num_agents)

        print("beliefs:", beliefs)

 
        if os.path.exists(backgrounds_file):
            self.backgrounds = self.load_backgrounds(backgrounds_file)
        else:
            self.backgrounds = self.create_and_save_backgrounds(self.num_agents, self.leaders, beliefs,
                                                                backgrounds_file)

        self.datacollector = DataCollector(
            model_reporters={
                "Polarization": self.compute_polarization,
                "NeighborCorrelationIndex": self.compute_nci,
                "EchoChamberEffect": self.compute_echo_chamber_effect,
            },
            agent_reporters={"Opinion": "opinion"}
        )     #  自动收集模型运行过程中每一步的数据

        for i, node in enumerate(tqdm(self.G.nodes(), desc="Creating agents")):
            agent = self.create_agent(i, beliefs)
            self.schedule.add(agent)
            self.grid.place_agent(agent, node)
            if node in leaders:
                self.leaders_pos[node] = agent.pos
                print(f"Pos of leader {node}: {agent.pos}")
        # 这段代码的作用是：为每个节点创建并初始化一个 Agent，让他们参与模型调度和网络结构，并记录下领导者的位置。

    def create_and_save_backgrounds(self, num_agents, leaders, beliefs, file_path):
        backgrounds = {}

        topics = self.opinion_templates

        print(topics)

        for i in tqdm(range(num_agents), desc="Initializing agents"):
            # 基础信息生成保持不变
            name = f"Agent_{i}"
            age = random.randint(18, 65)
            qualification = self.generate_qualification()
            traits = generate_big5_traits(1)
            gender = self.generate_gender()

            system_prompt_temp = system_prompt_template.format(
                name=name, gender=gender,
                age=age, traits=traits,
                qualification=qualification
            )

            # 修改数据结构以支持多话题
            backgrounds[str(i)] = {
                "name": name,
                "age": age,
                "education level": qualification,
                "traits": traits,
                "gender": gender,
                "system_prompt": None,
                "initial_opinions": {},  # 改为字典存储各话题观点
                "initial_reasonings": {}  # 改为字典存储各话题推理
            }

            # 获取当前agent的三个话题信念值
            print(type(beliefs))  # 检查是 list 还是 dict
            print(beliefs)        # 查看具体内容
            agent_beliefs = beliefs[str(i)]  # 例如 [1, -1, 0]

            # 为每个话题生成观点和推理
            leader_opinions = []  # 记录leader的所有话题观点
            for topic_idx, (topic_key, topic_question) in enumerate(topics.items()):
                belief = agent_beliefs[topic_idx]
                first_entry = {topic_key: topic_question}
                print("first_entry:",first_entry)
                
                # 生成单个话题的观点和推理
                opinion, reasoning = self.generate_initial_opinion_and_reasoning(
                    first_entry,
                    system_prompt_temp,
                    belief,  # 传入当前话题的信念值
                    self.gpt_model,
                )
                
                # 存储到字典
                backgrounds[str(i)]["initial_opinions"][topic_key] = opinion
                backgrounds[str(i)]["initial_reasonings"][topic_key] = reasoning
                
                # 如果是leader，收集所有观点用于构建prompt
                if i in leaders:
                    leader_opinions.append(f"{topic_question}: {opinion}")

            # 构建最终系统prompt
            system_prompt = system_prompt_temp
            if i in leaders:
                leader_stances = "\n".join(leader_opinions)
                system_prompt += f"""
                You are an information distributor. Your firm stances are:
                {leader_stances}
                You must consistently maintain these positions and refrain from adopting opposing views.
                """
                print(system_prompt)

            backgrounds[str(i)]["system_prompt"] = system_prompt

        # 保存到文件
        with open(file_path, "w") as file:
            json.dump({"backgrounds": backgrounds}, file, indent=4)

        return backgrounds

    def load_backgrounds(self, file_path):
        with open(file_path, "r") as file:
            data = json.load(file)
        return data["backgrounds"]

    def create_agent(self, i, beliefs):

        background_item = self.backgrounds[str(i)]
        
        name = background_item["name"]
        age = background_item["age"]
        qualification = background_item["education level"]
        traits = background_item["traits"]
        gender = background_item["gender"]
        initial_belief = beliefs[str(i)]
        initial_opinion = background_item["initial_opinions"]
        initial_reasoning = background_item["initial_reasonings"]
        system_prompt = background_item["system_prompt"]

        agent = SocialAgent(model=self,
                            unique_id=i,
                            name=name,
                            gender=gender,
                            age=age,
                            traits=traits,
                            qualification=qualification,
                            initial_belief=initial_belief,
                            belief_keywords=self.belief_keywords,
                            initial_opinion=initial_opinion,
                            initial_reasoning=initial_reasoning,
                            system_prompt=system_prompt,
                            gpt_model=self.gpt_model,
                            temp=self.temp,
                            mitigation_perspectives=None,
                            with_long_memory=self.with_long_memory,
                            topic=self.topic)

        return agent
    
    # 初始化三话题的信念值
    def generate_2topics_balanced_beliefs(self, num_agents, num_topics=2):
        belief_values = [-2, -1, 1, 2]
        
        # 生成每个话题的信念分布（保持原有的平衡逻辑）
        topic_beliefs = []
        for _ in range(num_topics):
            # 对每个话题独立应用原有的平衡生成逻辑
            beliefs_per_value = num_agents // len(belief_values)
            extra = num_agents % len(belief_values)
            
            beliefs = []
            for value in belief_values:
                beliefs.extend([value] * beliefs_per_value)
            
            if extra > 0:
                additional_beliefs = random.choices(belief_values, k=extra)
                beliefs.extend(additional_beliefs)
            
            random.shuffle(beliefs)  # 打乱顺序
            topic_beliefs.append(beliefs)
        
        # 将"话题优先"的列表转换为"智能体优先"的列表
        multi_topic_beliefs = [
            [topic_beliefs[topic][agent] for topic in range(num_topics)]
            for agent in range(num_agents)
        ]
        
        return multi_topic_beliefs
    
    def generate_2topics_balanced_beliefs1(self, num_agents, num_topics=2):
        belief_values = [-2, -1, 1, 2]
        
        # 生成一次平衡信念列表
        beliefs_per_value = num_agents // len(belief_values)
        extra = num_agents % len(belief_values)
        
        base_beliefs = []
        for value in belief_values:
            base_beliefs.extend([value] * beliefs_per_value)
        
        if extra > 0:
            base_beliefs.extend(random.choices(belief_values, k=extra))
        
        random.shuffle(base_beliefs)  # 打乱顺序一次

        # 将同一个信念列表复制给每个话题
        topic_beliefs = [base_beliefs.copy() for _ in range(num_topics)]
        
        # 转换为“智能体优先”列表
        multi_topic_beliefs = [
            [topic_beliefs[topic][agent] for topic in range(num_topics)]
            for agent in range(num_agents)
        ]
        
        return multi_topic_beliefs


    def generate_balanced_beliefs_dict(self, num_agents, num_topics=2):
        # 生成列表形式的多话题信念
        list_beliefs = self.generate_2topics_balanced_beliefs(num_agents, num_topics)
        
        # 转换为字典形式
        dict_beliefs = {
            str(agent_id): beliefs 
            for agent_id, beliefs in enumerate(list_beliefs)
        }
        
        return dict_beliefs
    
    def generate_balanced_beliefs_dict1(self, num_agents, num_topics=2):
        # 生成列表形式的多话题信念
        list_beliefs = self.generate_2topics_balanced_beliefs1(num_agents, num_topics)
        
        # 转换为字典形式
        dict_beliefs = {
            str(agent_id): beliefs 
            for agent_id, beliefs in enumerate(list_beliefs)
        }
        
        return dict_beliefs
    # 初始化三话题的信念值


    # 衡量回音室效应的方法
    def compute_echo_chamber_effect(self, i=None):

        print("i:", i)

        if i is None:
        # 递归调用，返回所有话题的 echo chamber 指数列表
            return [self.compute_echo_chamber_effect(j) for j in range(len(self.topics))]
        else:
            total_similarity = 0
            total_connections = 0

            for agent in self.schedule.agents:
                neighbors = self.grid.get_neighbors(agent.pos, include_center=False)
                agent_belief = agent.belief[i]

                for neighbor in neighbors:
                    neighbor_belief = neighbor.belief[i]
                    # similarity = 1 - abs(agent_belief - neighbor_belief)
                    similarity = 1 - abs(agent_belief - neighbor_belief) / 4
                    total_similarity += similarity
                    total_connections += 1

            if total_connections == 0:
                return 0

            echo_chamber_index = total_similarity / total_connections
            return echo_chamber_index


    def load_opinions(self):
        with open("opinions-r.json", "r") as file:
            return json.load(file)

    def load_belief_keywords_file(self, belief_keywords_file):
        with open(belief_keywords_file, "r") as file:
            keywords_file = json.load(file)
        return keywords_file
    
    # 生成初始观点和原因
    def generate_initial_opinion_and_reasoning(self, topic_key, system_prompt, initial_belief, gpt_model):

        belief_str = str(initial_belief)
        # with open(self.belief_keywords_file, "r") as file:
        #     keyword_file = json.load(file)
        belief_keywords = self.belief_keywords
        system_msg = system_prompt

        user_msg = init_opinion_prompt.format(topic = topic_key, belief=initial_belief,
                                              keyword=random.choice(belief_keywords[belief_str]))

        # initial temp is high.
        response = get_completion_from_messages_structured(system_messages=system_msg, messages=user_msg,
                                                           model=gpt_model,
                                                           temperature=0.7, response_type=update_opinion_response)
        
        print("Response raw:", response)

        if response is None:
            raise ValueError("GPT 返回了 None, 请检查 API 调用或参数")

        print("response:", response)

        tweet = response.opinion
        reasoning = response.reasoning

        return tweet, reasoning

    def generate_qualification(self):
        qualifications = ["No Education", "High School", "Bachelor's Degree", "Master's Degree", "PhD"]
        return random.choice(qualifications)

    def generate_traits(self):
        traits = ["Extroverted", "Introverted", "Cautious", "Risk-taking", "Analytical", "Emotional", "Assertive",
                  "Flexible"]
        return random.choice(traits)

    def generate_gender(self):
        genders = ['male', 'female']
        return random.choice(genders)
    
    def save_network_structure(self):
        clustering_coefficient = nx.average_clustering(self.G)
        avg_path_length = nx.average_shortest_path_length(self.G) if nx.is_connected(self.G) else None
        density = nx.density(self.G)
        diameter = nx.diameter(self.G) if nx.is_connected(self.G) else None

        network_data = {
            "nodes": list(self.G.nodes),
            "edges": list(self.G.edges),
            "clustering_coefficient": clustering_coefficient,
            "average_path_length": avg_path_length,
            "density": density,
            "diameter": diameter
        }

        file_path = os.path.join(self.run_dir, "network_structure.json")
        with open(file_path, "w") as file:
            json.dump(network_data, file, indent=4)

        plt.figure(figsize=(8, 8))
        nx.draw(self.G, with_labels=True, node_color='skyblue', node_size=500, edge_color='gray', font_size=10)
        plt.title(f"Network Structure for {self.name}")
        plt_path = os.path.join(self.run_dir, "network_structure.png")
        plt.savefig(plt_path)
        plt.close()

    def save_agents_data(self, file_path):
        agents_data = {}

        for agent in self.schedule.agents:
            agents_data[agent.unique_id] = {
                "opinions": agent.opinions,
                "beliefs": agent.beliefs,
                "reasonings": agent.reasonings,
                "short-memory": agent.short_memory_full,
                "long_memory": agent.long_memory_full,
            }

        with open(file_path, "w") as file:
            json.dump(agents_data, file, indent=4)

    
    def save_model_data(self):
        topic_data = {}
        for i, topic in enumerate(self.topics):
            
            polarizations = self.compute_polarization(i)
            nci = self.compute_nci(i)
            echo = self.compute_echo_chamber_effect(i)

            topic_data[topic] = {
                "polarization": polarizations,
                "neighbor_correlation_index": nci,
                "echo_chamber_index": echo
            }

        # 构造最终模型数据结构
        model_data = {
            "step": self.schedule.time,
            "date": str(self.current_date),
            "topics": topic_data
        }

        file_path = os.path.join(self.run_dir, "model_overview.json")
        with open(file_path, "a") as file:
            json.dump(model_data, file)
            file.write("\n")

    def compute_polarization(self, i=None):

        if i is None:
        # 计算所有话题的 polarization，返回列表
            return [self.compute_polarization(j) for j in range(len(self.topics))]
        else:
            beliefs = [agent.belief[i] for agent in self.schedule.agents]
            print("beliefs:",beliefs)
            polarization_index = sum(beliefs) / len(beliefs)
            return polarization_index

    def compute_nci(self, i=None):
        if i is None:
            return [self.compute_nci(j) for j in range(len(self.topics))]
        else:
            return sum([abs(agent.belief[i] - neighbor.belief[i]) < 0.1 for agent in self.schedule.agents
                        for neighbor in self.grid.get_neighbors(agent.pos)]) / self.num_agents
        
    def decide_agent_interactions(self, recommendation='random'):

        print(recommendation)

        all_agents_reasons = {}
        
        for agent in self.schedule.agents:

            neighbors = self.grid.get_neighbors(agent.pos)
            agent.agent_interaction = []

            # 可调参数：最大交互人数（如无设置，默认10）
            k = self.max_interactions if self.max_interactions > 0 else len(neighbors)

            # To do: more recommendation algorithms
            if recommendation == 'random':
                random.shuffle(neighbors)
                neighbors = neighbors[:self.max_interactions]
            elif recommendation == 'similarity':
                neighbors_selected = []
                for neighbor in neighbors:
                    if abs(agent.belief - neighbor.belief) <= 2:
                        neighbors_selected.append(neighbor)
                neighbors = neighbors_selected
            elif recommendation == 'dissimilarity':
                neighbors_selected = []
                for neighbor in neighbors:
                    if abs(agent.belief - neighbor.belief) >= 2:
                        neighbors_selected.append(neighbor)
                neighbors = neighbors_selected
            elif recommendation == 'average_based':
                epsilon = 2
                neighbors_selected = []
                
                # Step 1: 计算当前 agent 的平均观点
                agent_avg = sum(agent.belief) / len(agent.belief)
                
                # Step 2: 遍历所有邻居，计算他们的平均观点并比较
                for neighbor in neighbors:
                    neighbor_avg = sum(neighbor.belief) / len(neighbor.belief)
                    if abs(agent_avg - neighbor_avg) <= epsilon:  # epsilon 是置信区间
                        neighbors_selected.append(neighbor)

                neighbors = neighbors_selected

            elif recommendation == 'prompt_choose':
                neighbors_selected = []
                neighbor_is_choose = []

                agent_reasons = []  # 新增：该agent的所有邻居决策
                for neighbor in neighbors:
                    user_msg = recommendate_interact2.format(age=agent.age, gender=agent.gender, qualification=agent.qualification, traits=agent.traits,
                                                            age1=neighbor.age, gender1=neighbor.gender, qualification1=neighbor.qualification, traits1=neighbor.traits,
                                                            agent_topic_opinions=agent.belief, neighbor_topic_opinions=neighbor.belief)
                    # print(user_msg)
                    response = get_completion_from_messages_structured(system_messages=agent.system_prompt, messages=user_msg,
                                                           model=agent.gpt_model,
                                                           temperature=self.temp, response_type=recommendate_interact1)
                    # print("raw_response:", raw_response)
                    
                    # response = self.extract_json_only(raw_response)
                    print("response:",response)

                    neighbor_is_choose.append(response.Output)

                    # 保存每个邻居的决策和理由
                    agent_reasons.append({
                        "neighbor_id": neighbor.name,
                        "decision": response.Output,
                        "reason": response.reason
                    })

                    print(agent_reasons)

                    if response.Output == "Yes":
                        
                        neighbors_selected.append(neighbor)
                neighbors = neighbors_selected
                print(neighbor_is_choose)

                all_agents_reasons[agent.name] = agent_reasons  # 收集该agent决策数据

                # 加reason思考。
            
            # 设置prompt提示。
                # print("neighbors:", neighbors)

            for neighbor in neighbors:

                neighbor_agent = neighbor
                agent.agent_interaction.append(neighbor_agent)

        # self.save_all_agents_reasons(all_agents_reasons, save_dir="data1/interact_reason")
            

    def step(self):
        self.decide_agent_interactions(recommendation=self.recommendation)
        print(f"start step: {self.schedule.time}")
    

        if self.schedule.time == self.mitigation_step:
            polarizations = self.compute_polarization(0)
            nci = self.compute_nci(0)
            echo = self.compute_echo_chamber_effect(0)

            topic_msg = generate_topic_prompt.format(current_topic=self.topic,polarization=polarizations,nci=nci,echo=echo)
            # print(topic_msg)

            response = get_completion_from_messages_structured(messages=topic_msg,model=self.gpt_model,temperature=self.temp, response_type=generate_topic)
            print(response)

            # agent.mitigation_perspectives = response.topic

            # 保存目录
            save_dir = "./saved_topics"
            os.makedirs(save_dir, exist_ok=True)  # 不存在就创建目录

            save_path = os.path.join(save_dir, "mitigation_topics.json")

            # 读取已有数据
            if os.path.exists(save_path):
                with open(save_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = []

            # 新增记录
            new_record = {
                "time_step": self.schedule.time,
                "original_topic": self.topic1,
                "polarization": polarizations,
                "nci": nci,
                "echo": echo,
                "mitigation_topic": response.topic
            }
            data.append(new_record)

            event_b = response.topic
            print("event_b:",event_b)

            # 写回文件
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
                        # 修改 opinions-m.json
            save_path1 = "./opinions-m.json"

            with open(save_path1, "r", encoding="utf-8") as f:
                data1 = json.load(f)

            # 保证有序，拿到第二个 key
            keys = list(data1.keys())
            if len(keys) >= 2:
                second_key = keys[1]          # 只找第二个 key
                data1[second_key] = response.topic  # 只更新 value，不改 key

            with open(save_path1, "w", encoding="utf-8") as f:
                json.dump(data1, f, ensure_ascii=False, indent=2)

            for agent in self.schedule.agents:
                # print(agent.belief)
                # print(agent.opinion)
                # print(agent.reasoning)
                opinion_a = list(agent.opinion.values())[0]
                reasoning_a = list(agent.reasoning.values())[0]
                mig_msg = init_mig_opinion_prompt.format(event_a=self.topic, belief_a=agent.belief[0], opinion_a=opinion_a, 
                                                         reasoning_a=reasoning_a, event_b=event_b, belief_keywords=self.belief_keywords)
                # print(mig_msg)

                response = get_completion_from_messages_structured(messages=mig_msg,model=self.gpt_model,
                                                                   temperature=self.temp, response_type=init_mig_opinion_prompt1)
                print(response)

                agent.belief[0] = response.belief
                agent.opinion[0] = response.opinion
                agent.reasoning[0] = response.reasoning


        self.schedule.step()
        self.save_all_agents_topic_history("data1/choice-r/choose_topics1.json")

        for agent in self.schedule.agents:
            # agent.mitigation_perspectives = None
            update_day(agent)

        self.datacollector.collect(self)
        self.current_date += timedelta(days=1)

        agents_file_path = os.path.join(self.run_dir, f"agents_interaction_data.json")
        self.save_agents_data(agents_file_path)

        self.save_model_data()

    def run_model(self, step_count):
        for _ in tqdm(range(step_count), desc="Running Model"):
            self.step()

            print(
                f"Current date: {self.current_date}, Polarization: {self.compute_polarization()}, Echo Chamber Effect: {self.compute_echo_chamber_effect()}")

        self.save_checkpoint(os.path.join(self.run_dir, f"{self.name}_checkpoint.pkl"))
        agents_file_path = os.path.join(self.run_dir, "agents_data.json")
        self.save_agents_data(agents_file_path)

    def save_checkpoint(self, file_path):
        with open(file_path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load_checkpoint(file_path):

        with open(file_path, "rb") as file:
            return pickle.load(file)
    
    def save_all_agents_topic_history(self, filepath):
        data = {}
        for agent in self.schedule.agents:
            data[agent.name] = agent.selected_topics_history
        with open(filepath, "w") as f:
            json.dump(data, f)

    def extract_json_only(self, text):
        if not isinstance(text, str):
            raise ValueError(f"Expected string from LLM, got: {type(text)}")

        # 尝试提取最前面的 JSON 块（防止后面多余解释）
        match = re.search(r'\{[\s\S]*?\}', text)
        if match:
            json_str = match.group()
            return json.loads(json_str)
        
        raise ValueError(f"No valid JSON found in:\n{text}")
    
    def save_all_agents_reasons(self, all_agents_reasons, save_dir="data/agents_reasons"):
        """
        保存所有agent当前轮次对邻居交互决策的reason信息到JSON文件。
        
        参数:
        - all_agents_reasons: dict, key是agent id，value是邻居决策列表
        - save_dir: str, 保存目录路径，默认data/agents_reasons
        """
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"agent_neighbors_reasons_step_{self.schedule.time}_{timestamp}.json"
        save_path = os.path.join(save_dir, filename)
        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(all_agents_reasons, f, ensure_ascii=False, indent=4)
        
        print(f"Saved all agents reasons to {save_path}")