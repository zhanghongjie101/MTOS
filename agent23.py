import json
import math
from mesa import Agent
from utils3 import *
from prompt2 import *
from collections import defaultdict
from collections import Counter

def compute_fatigue_level(tsr: float, b: float = 3.0) -> float:
    """
    根据话题选择率(TSR)计算疲劳度(Fatigue Level)

    参数:
        tsr (float): 话题选择率，范围为 [0, 1]
        b (float): 增长速率控制参数，越大表示疲劳增长越陡(默认 b = 3.0)

    返回:
        float: 疲劳度，范围为 [0, 1]
    """
    if not 0 <= tsr <= 1:
        raise ValueError("话题选择率 tsr 必须在 0 到 1 之间。")

    # 指数函数映射：归一化处理使得最大疲劳度为 1
    numerator = math.exp(b * tsr) - 1
    denominator = math.exp(b) - 1
    fatigue_level = numerator / denominator

    return round(fatigue_level, 4)  # 保留4位小数

def get_summary_long(system_prompt, long_memory, short_memory, gpt_model, temp=0.5):

    system_msg = system_prompt
    user_msg = long_memory_prompt.format(long_memory=long_memory, short_memory=short_memory)

    get_summary = get_completion_from_messages_structured(system_messages=system_msg, messages=user_msg,
                                                          model=gpt_model,
                                                          temperature=temp, response_type=long_memory_response).long_term_memory
    # print('long_memory: ', get_summary)

    return get_summary


def get_summary_short(system_prompt, opinions, topic_opinions, topic_opinions_count, gpt_model, topic ,mitigation_perspectives=None, temp=0.5):

    opinions_text = "\n".join(f"One of your close contacts believes: {opinion}" for opinion in opinions)
    current_topic = topic

    if mitigation_perspectives is not None:
        random.shuffle(mitigation_perspectives)

        for i in range(len(opinions) // 2 + 1):
            opinions_text += f"\n You heard that: {random.sample(mitigation_perspectives, 1)}"

    # user_msg = reflecting_prompt.format(opinions=opinions_text, topic=topic)
    system_msg = system_prompt
    user_msg = reflecting_prompt.format(current_topic=current_topic , opinions=opinions_text, 
                                        topic_opinions=topic_opinions, topic_opinions_count=topic_opinions_count)
    # print("user_msg:",user_msg)

    # 这里的反思prompt需要修改

    # print("short term usr message: ", user_msg)

   # msg = [{"role": "user", "content": user_msg}]

    get_summary = get_completion_from_messages_structured(system_messages=system_msg, messages=user_msg,
                                                          model=gpt_model,
                                                          temperature=temp, response_type=reflecting_response).short_term_memory
    # print('short_memory: ', get_summary)

    return get_summary


class SocialAgent(Agent):
    def __init__(self, model, unique_id, name, gender, age, traits, qualification, initial_belief, topic,
                 belief_keywords, gpt_model, temp=0.5, initial_opinion=None, initial_reasoning=None,
                 with_long_memory=True, mitigation_perspectives=None,
                 system_prompt="You are a helpful assistant"):
        super().__init__(unique_id, model)

        self.model = model
        self.name = name
        self.gender = gender
        self.age = age
        self.traits = traits
        self.qualification = qualification
        self.topic = topic
        self.belief_keywords = belief_keywords
        self.system_prompt = system_prompt

        self.with_long_memory = with_long_memory

        self.temp = temp
        self.gpt_model = gpt_model

        self.mitigation_perspectives = mitigation_perspectives


        self.initial_opinion = initial_opinion
        self.opinions = [self.initial_opinion]
        self.initial_belief = initial_belief
        self.belief = initial_belief
        self.beliefs = [self.belief]


        self.short_memory_full = []

        self.long_opinion_memory = ""
        self.long_memory_full = [self.long_opinion_memory]

        self.initial_reasoning = initial_reasoning
        self.reasonings = [initial_reasoning]

        self.agent_interaction = []
        self.contact_ids = []

        self.selected_topics_history = []  # 新增这个属性，用来存储所有选的话题,该agent所选择的话题历史

        # 给agent也赋予话题属性，便于调用
        self.opinion_templates = self.load_opinions1()
        # self.topic = self.opinion_templates[topic]
        # 引入三个话题
        self.topics = list(self.opinion_templates.keys())

    def interact(self):
        print("I'm agent ", self.unique_id)
        others_opinions = []
        contact_id = []
        topic_opinions = {}
        topic_opinions_count = {} # 按话题统计数量
        contact_id = []

        # 统计所有Agent的所有选择话题列表
        all_topics = []
        candidate_topics = self.topics  # 可根据实际候选话题调整
        try:
            with open('data1/choice-m/choose_topics.json', 'r', encoding='utf-8') as f:
                history_data = json.load(f)
            for agent_choices in history_data.values():
                all_topics.extend(agent_choices)

            # 统计整体话题频率(话题热度)
            if all_topics:
                total_choices = len(all_topics)
                topic_counts = Counter(all_topics)
                topic_heat = {topic: topic_counts.get(topic, 0) / total_choices for topic in candidate_topics}
            else:
                topic_heat = {topic: 0.0 for topic in candidate_topics}

            # 统计该Agent的话题选择率

            # 疲劳度和选择率成指数级别变化。不能对等。

            agent_choices = history_data.get(self.name, [])
            total_agent_choices = len(agent_choices)

            if total_agent_choices == 0:
                agent_topic_choise = {topic: 0.0 for topic in candidate_topics}
            else:
                agent_counts = Counter(agent_choices)
                agent_topic_choise = {
                    topic: agent_counts.get(topic, 0) / total_agent_choices for topic in candidate_topics
                }

        except FileNotFoundError:
            # 若找不到文件，则所有话题热度和疲劳度均为0
            topic_heat = {topic: 0.0 for topic in candidate_topics}
            agent_topic_choise = {topic: 0.0 for topic in candidate_topics}
        
        # 计算对应疲劳度(Fatigue Level)
        agent_topic_fatigue = {
            topic: compute_fatigue_level(tsr, b=5.0)
            for topic, tsr in agent_topic_choise.items()
        }
          
        # print("topic_heat:",topic_heat)
        # print("agent_topic_fatigue:",agent_topic_fatigue)

        topic_values = [self.opinion_templates[t] for t in self.topics]

        # print("Lenth of interaction is:", len(self.agent_interaction))
        for agent in self.agent_interaction:

            if len(set(topic_values)) == 1:
                # 两个话题相同，直接选第一个，不保存历史
                choose_topic = self.topics[0]
            else:
                # 不同话题 → GPT 推荐逻辑
                topics_str = ", ".join(self.topics)
                user_msg2 = recommendate_topic_prompt1.format(
                    age=self.age,
                    gender=self.gender,
                    qualification=self.qualification,
                    traits=self.traits,
                    topics=topics_str,
                    memory=self.long_memory_full[-1],
                    topic_heat=topic_heat,
                    topic_fatigue=agent_topic_fatigue
                )

                choose_topic = self.select_topic(user_msg2, self.gpt_model)

                # 只有不同话题时才保存选择结果
                self.selected_topics_history.append(choose_topic)
            

            contact_id.append(agent.unique_id)
            agent_latest_opinion = agent.opinions[-1]

            interact_opinion = agent_latest_opinion[choose_topic]
            # print("interact_opinion:", interact_opinion)
            
            others_opinions.append(interact_opinion)

            # 收集：按话题分类
            if choose_topic not in topic_opinions:
                topic_opinions[choose_topic] = []
            topic_opinions[choose_topic].append(interact_opinion)

                # 按话题统计数量
            if choose_topic not in topic_opinions_count:
                topic_opinions_count[choose_topic] = 0
            topic_opinions_count[choose_topic] += 1

        
        
        # self.short_opinion_memory.append(others_opinions)
        self.contact_ids.append(contact_id)

        opinion_short_summary = get_summary_short(self.system_prompt, others_opinions, topic_opinions=topic_opinions, topic_opinions_count=topic_opinions_count,
                                                  gpt_model=self.gpt_model,topic=self.topic,
                                                  mitigation_perspectives=self.mitigation_perspectives,
                                                  temp=self.temp)
        # 这里传入了topic=self.topic

        self.short_memory_full.append(opinion_short_summary)

        if self.with_long_memory:
            long_mem = get_summary_long(self.system_prompt, self.long_opinion_memory, opinion_short_summary,
                                        gpt_model=self.gpt_model,
                                        temp=self.temp)

            self.long_opinion_memory = long_mem
            self.long_memory_full.append(self.long_opinion_memory)

        self.agent_interaction = []



    def response_and_belief(self, user_msg, gpt_model):

        system_msg = self.system_prompt

        response = get_completion_from_messages_structured(system_messages=system_msg, messages=user_msg,
                                                           model=gpt_model,
                                                           temperature=self.temp, response_type=update_belief_response)
        

        print("response:",response)

        # tweet = response.opinion
        belief = response.belief
        # reasoning = response.reasoning

        return belief
    
    
    def response_and_opinion(self, user_msg, gpt_model):

        system_msg = self.system_prompt

        response = get_completion_from_messages_structured(system_messages=system_msg, messages=user_msg,
                                                           model=gpt_model,
                                                           temperature=self.temp, response_type=update_opinion_response)

        tweet = response.opinion
        # belief = response.belief
        reasoning = response.reasoning

        return tweet, reasoning
    
    def select_topic(self, prompt_messages, gpt_model):

        system_msg = self.system_prompt

        response = get_completion_from_messages_structured(
            system_messages=system_msg,  
            messages=prompt_messages,
            model=gpt_model,
            temperature=self.temp,
            response_type=select_topic1  # 如果你有 dataclass 结构，可以替换为对应类型
        )

        # print("LLM Response:", response)

        selected_topic = response.topic

        return selected_topic  # 如果需要，可返回 selected_topic, reason

    
    def load_opinions1(self):
        with open("opinions-r.json", "r") as file:
            return json.load(file)

    def step(self):
        '''
        Step function for agent
        '''
        self.interact()

