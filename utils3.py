#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from names_dataset import NameDataset
import numpy as np
import random
#from openai import OpenAI
import openai
import json
from pydantic import BaseModel
from prompt2 import *
import time
import requests
import os
import shutil

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
import seaborn as sns
from scipy.stats import pearsonr


client = openai.Client(
    # This is the default and can be omitted
    api_key="xxxx",
    base_url="xxx"
)


# 计算图中节点意见与邻居平均意见的皮尔逊相关系数（NCI，邻居相关指数）。
def metric_neighbors_correlation_index(G, opinions):
    """
    Calculate the Neighbors Correlation Index (NCI) for a graph G based on opinions.

    Parameters:
    G: networkx.Graph
        The input graph where nodes are individuals and edges represent connections.
    opinions: dict
        A dictionary of opinions where keys are node ids and values are the opinions (float).

    Returns:
    nci: float
        The Neighbors Correlation Index for the graph.
    """
    nodes = list(G.nodes())

    node_opinions = np.array([opinions[node] for node in nodes])

    neighbor_avg_opinions = []

    for node in nodes:
        neighbors = list(G.neighbors(node))
        if len(neighbors) > 0:
            avg_opinion = np.mean([opinions[neighbor] for neighbor in neighbors])
        else:
            avg_opinion = opinions[node]

        neighbor_avg_opinions.append(avg_opinion)

    neighbor_avg_opinions = np.array(neighbor_avg_opinions)

    nci, _ = pearsonr(node_opinions, neighbor_avg_opinions)

    return nci

# 计算网络整体意见极化程度（意见值离均值的平方和）。
def metric_polarization(G, opinions):

    opinion_values = list(opinions.values())

    n = len(opinion_values)

    mean_opinion = np.mean(opinion_values)

    polarization = np.sum((np.array(opinion_values) - mean_opinion) ** 2)

    return polarization

# 计算网络中邻居间意见的加权平方差的总和。
def metric_global_disagreement(G, opinions):

    global_disagreement = 0

    for i in G.nodes():
        local_disagreement = 0
        for j in G.neighbors(i):
            weight_ij = G[i][j].get('weight', 1)
            opinion_diff = opinions[i] - opinions[j]
            local_disagreement += weight_ij * (opinion_diff ** 2)

        global_disagreement += local_disagreement

    global_disagreement *= 0.5

    return global_disagreement

# 定义了用于解析LLM响应的结构化数据模型。
# 方便对LLM返回的JSON响应进行类型校验和自动解析。
class update_belief_response(BaseModel):
    belief: float
class update_opinion_response(BaseModel):
    opinion: str
    reasoning: str
class reflecting_response(BaseModel):
    short_term_memory: str
class long_memory_response(BaseModel):
    long_term_memory: str
class select_topic1(BaseModel):
    topic:str
class recommendate_interact1(BaseModel):
    Output:str
    reason: str
class generate_topic(BaseModel):
    topic:str
class init_mig_opinion_prompt1(BaseModel):
    opinion: str
    belief: int
    reasoning: str


# 根据给定阈值返回布尔值，模拟概率事件是否发生。
def probability_threshold(threshold):
    '''
    Used in self.infect_interaction()
    '''
    # Generates random number from 0 to 1

    return (np.random.rand() < threshold)

# 代理人属性：模拟代理人的教育背景。
def generate_qualifications(n: int):
    '''
    Returns a list of random educational qualifications.

    Parameters:
    n (int): The number of qualifications to generate.
    '''

    # Define a list of possible qualifications including lower levels and no education
    qualifications = ['No Education', 'Primary School', 'Middle School',
                      'High School Diploma', 'Associate Degree', 'Bachelor\'s Degree',
                      'Master\'s Degree', 'PhD', 'Professional Certificate']

    # Randomly select n qualifications from the list
    generated_qualifications = random.choices(qualifications, k=n)

    return generated_qualifications


def generate_names(n: int, s: int, country_alpha2='US'):
    '''
    Returns random names as names for agents from top names in the USA
    Used in World.init to initialize agents
    '''

    # This function will randomly selct n names (n/2 male and n/2 female) without
    # replacement from the s most popular names in the country defined by country_alpha2
    if n % 2 == 1:
        n += 1
    if s % 2 == 1:
        s += 1

    nd = NameDataset()
    male_names = nd.get_top_names(s // 2, 'Male', country_alpha2)[country_alpha2]['M']
    female_names = nd.get_top_names(s // 2, 'Female', country_alpha2)[country_alpha2]['F']
    if s < n:
        raise ValueError(f"Cannot generate {n} unique names from a list of {s} names.")
    # generate names without repetition
    names = random.sample(male_names, k=n // 2) + random.sample(female_names, k=n // 2)
    del male_names
    del female_names
    random.shuffle(names)
    return names


def generate_big5_traits(n: int):
    '''
    Return big 5 traits for each agent
    Used in World.init to initialize agents
    '''

    # Trait generation
    agreeableness_pos = ['Cooperation', 'Amiability', 'Empathy', 'Leniency', 'Courtesy', 'Generosity', 'Flexibility',
                         'Modesty', 'Morality', 'Warmth', 'Earthiness', 'Naturalness']
    agreeableness_neg = ['Belligerence', 'Overcriticalness', 'Bossiness', 'Rudeness', 'Cruelty', 'Pomposity',
                         'Irritability',
                         'Conceit', 'Stubbornness', 'Distrust', 'Selfishness', 'Callousness']
    # Did not use Surliness, Cunning, Predjudice,Unfriendliness,Volatility, Stinginess

    conscientiousness_pos = ['Organization', 'Efficiency', 'Dependability', 'Precision', 'Persistence', 'Caution',
                             'Punctuality',
                             'Punctuality', 'Decisiveness', 'Dignity']
    # Did not use Predictability, Thrift, Conventionality, Logic
    conscientiousness_neg = ['Disorganization', 'Negligence', 'Inconsistency', 'Forgetfulness', 'Recklessness',
                             'Aimlessness',
                             'Sloth', 'Indecisiveness', 'Frivolity', 'Nonconformity']

    surgency_pos = ['Spirit', 'Gregariousness', 'Playfulness', 'Expressiveness', 'Spontaneity', 'Optimism', 'Candor']
    # Did not use Humor, Self-esteem, Courage, Animation, Assertion, Talkativeness, Energy level, Unrestraint
    surgency_neg = ['Pessimism', 'Lethargy', 'Passivity', 'Unaggressiveness', 'Inhibition', 'Reserve', 'Aloofness']
    # Did not use Shyness, Silenece

    emotional_stability_pos = ['Placidity', 'Independence']
    emotional_stability_neg = ['Insecurity', 'Emotionality']
    # Did not use Fear, Instability, Envy, Gullibility, Intrusiveness

    intellect_pos = ['Intellectuality', 'Depth', 'Insight', 'Intelligence']
    # Did not use Creativity, Curiousity, Sophistication
    intellect_neg = ['Shallowness', 'Unimaginativeness', 'Imperceptiveness', 'Stupidity']

    # Combine each trait
    agreeableness_tot = agreeableness_pos + agreeableness_neg
    conscientiousness_tot = conscientiousness_pos + conscientiousness_neg
    surgency_tot = surgency_pos + surgency_neg
    emotional_stability_tot = emotional_stability_pos + emotional_stability_neg
    intellect_tot = intellect_pos + intellect_neg

    # create traits list to be returned
    traits_list = []

    for _ in range(n):
        agreeableness_rand = random.choice(agreeableness_tot)
        conscientiousness_rand = random.choice(conscientiousness_tot)
        surgency_rand = random.choice(surgency_tot)
        emotional_stability_rand = random.choice(emotional_stability_tot)
        intellect_rand = random.choice(intellect_tot)

        selected_traits = [agreeableness_rand, conscientiousness_rand, surgency_rand,
                           emotional_stability_rand, intellect_rand]

        traits_chosen = (', '.join(selected_traits))
        traits_list.append(traits_chosen)
    del agreeableness_rand
    del conscientiousness_rand
    del surgency_rand
    del emotional_stability_rand
    del intellect_rand
    del selected_traits
    del traits_chosen
    return traits_list

# 将数字n分解成两个乘数，用于布局代理人网格。
def factorize(n):
    '''
    Factorize number for ideal grid dimensions for # of agents
    Used in World.init
    '''
    for i in range(int(n ** 0.5), 1, -1):
        if n % i == 0:
            return (i, n // i)
    return (n, 1)


def decay_opinion(opinion_value, decay_rate=0.01):
    """
    对意见值进行持续性、细微的指数衰减（适用于 [-2, 2] 范围）

    参数:
        opinion_value (float): 当前个体的意见值，范围在 [-2, 2]
        decay_rate (float): 衰减率 λ，值越小衰减越细微，建议 0.001 ~ 0.05

    返回:
        float: 衰减后的意见值
    """
    decay_factor = np.exp(-decay_rate * abs(opinion_value))
    return opinion_value * decay_factor


def get_completion_from_messages_structured(messages, system_messages="You are a helpful assistant.",
                                            model="qwen2:1.5b", temperature=1, response_type=None):
    success = False
    retry = 0
    max_retries = 30
    url = "http://localhost:11434/api/chat"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_messages},
            {"role": "user", "content": messages}
        ],
        "temperature": temperature,
        "stream": False  # 你也可以设成 True 来逐步生成
    }

    while retry < max_retries and not success:
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            result = response.json()

            # 获取 LLM 返回内容
            content = result["message"]["content"]

            # 如果你有 response_type（即结构化输出格式），需要你自己解析
            if response_type:
                # 模拟 structured parse（这里你可能要用 Pydantic 或正则）
                try:
                    parsed = response_type.parse_raw(content)
                    return parsed
                except Exception as parse_err:
                    print(f"Parse Error: {parse_err}\nRaw content: {content}")
                    return None
            else:
                return content

        except Exception as e:
            print(f"Error: {e}\nRetrying...")
            retry += 1
            time.sleep(2)

    return None


# 驱动代理每天（每步）基于当前记忆和观点，向LLM发送请求，获取新意见。
def update_day1(agent):

    topics = agent.topics
    # print("topics:",topics)

    new_opinions = agent.opinions[-1].copy()
    new_beliefs =  agent.beliefs[-1].copy()
    new_reasonings = agent.reasonings[-1].copy()


    for index, topic in enumerate(topics):
        # print(f"Topic: {topic}")
        latest_opinion = agent.opinions[-1]
        topic_opinion1 = latest_opinion[topic]
        # print("topic_opinion:", topic_opinion1)
        topic_belief1 = agent.belief[index]
        # print("topic_belief:", topic_belief1)

        user_msg = update_belief_prompt.format(
                                            belief=topic_belief1,
                                            long_mem=agent.long_opinion_memory,
                                            topic=topic,
                                            current_topic=agent.topic,
                                            opinion=topic_opinion1)
        # print(user_msg)
        new_belief= agent.response_and_belief(user_msg, agent.gpt_model)

        # 信念值缓慢下降，衰减规律
        new_belief = decay_opinion(new_belief, decay_rate=0.01)
        # print("new_belief_decay:",new_belief)
        user_msg1 = update_opinion_prompt.format(belief_keywords=agent.belief_keywords, belief=new_belief, topic=topic)
        # print(user_msg1)
        new_opinion, new_reasoning = agent.response_and_opinion(user_msg1, agent.gpt_model)

        # print("new_opinion, new_reasoning:",new_opinion, new_reasoning)

        # new_opinion, new_reasoning = agent.response_and_belief

        # 更新Topic的最新观点
        new_opinions[topic] = new_opinion
        # print("agent.opinion:", new_opinions)

        # 更新Topic的最新信念
        new_beliefs[index] = new_belief
        # print("agent.belief:", new_beliefs)

        # 更新Topic的最新原因
        new_reasonings[topic] = new_reasoning
        # print("agent.reasoning", new_reasonings)
    
    agent.opinion = new_opinions
    agent.belief = new_beliefs
    agent.reasoning = new_reasonings
    
    # print("new_opinions:", new_opinions)
    # print("new_beliefs:", new_beliefs)
    # print("new_reasonings:", new_reasonings)
    
    agent.opinions.append(new_opinions)
    agent.beliefs.append(new_beliefs)
    agent.reasonings.append(new_reasonings)

def update_day(agent):

    topics = agent.topics
    topics_template = agent.opinion_templates
    topic_values = [agent.opinion_templates[t] for t in agent.topics]
    # print("topics:", topics)

    new_opinions = agent.opinions[-1].copy()
    new_beliefs = agent.beliefs[-1].copy()
    new_reasonings = agent.reasonings[-1].copy()

    if len(set(topic_values)) == 1:
        # 如果所有话题都一样，只计算一次，结果复制给所有话题
        topic = topics[0]
        latest_opinion = agent.opinions[-1]
        topic_opinion1 = latest_opinion[topic]
        topic_belief1 = agent.belief[0]

        user_msg = update_belief_prompt.format(
            belief=topic_belief1,
            long_mem=agent.long_opinion_memory,
            topic=topic,
            current_topic=agent.topic,
            opinion=topic_opinion1
        )

        new_belief = agent.response_and_belief(user_msg, agent.gpt_model)

        # 信念值衰减
        new_belief = decay_opinion(new_belief, decay_rate=0.01)

        user_msg1 = update_opinion_prompt.format(
            belief_keywords=agent.belief_keywords,
            belief=new_belief,
            topic=topic
        )
        new_opinion, new_reasoning = agent.response_and_opinion(user_msg1, agent.gpt_model)

        # 把结果赋值给所有话题
        for index in range(len(topics)):
            new_opinions[topics[index]] = new_opinion
            new_beliefs[index] = new_belief
            new_reasonings[topics[index]] = new_reasoning

    else:
        # 话题不同，逐个计算
        for index, topic in enumerate(topics):
            latest_opinion = agent.opinions[-1]
            topic_opinion1 = latest_opinion[topic]
            topic_belief1 = agent.belief[index]

            user_msg = update_belief_prompt.format(
                belief=topic_belief1,
                long_mem=agent.long_opinion_memory,
                topic=topic,
                current_topic=agent.topic,
                opinion=topic_opinion1
            )
            new_belief = agent.response_and_belief(user_msg, agent.gpt_model)

            # 信念值衰减
            new_belief = decay_opinion(new_belief, decay_rate=0.01)

            user_msg1 = update_opinion_prompt.format(
                belief_keywords=agent.belief_keywords,
                belief=new_belief,
                topic=topic
            )
            new_opinion, new_reasoning = agent.response_and_opinion(user_msg1, agent.gpt_model)

            new_opinions[topic] = new_opinion
            new_beliefs[index] = new_belief
            new_reasonings[topic] = new_reasoning

    # 更新 agent 状态
    agent.opinion = new_opinions
    agent.belief = new_beliefs
    agent.reasoning = new_reasonings

    agent.opinions.append(new_opinions)
    agent.beliefs.append(new_beliefs)
    agent.reasonings.append(new_reasonings)

# 从文件中读取所有代理人在所有时间步的信念数据，转换成时间步索引字典，方便后续分析和可视化。
def extract_beliefs_at_all_steps(agent_data_file_path):
    with open(agent_data_file_path, "r") as file:
        agent_data = json.load(file)

    time_steps = len(next(iter(agent_data.values()))["beliefs"])  # 获取时间步总数
    beliefs_at_steps = {time_step: {} for time_step in range(time_steps)}

    for agent_id, agent_info in agent_data.items():
        for time_step in range(time_steps):
            beliefs_at_steps[time_step][agent_id] = agent_info["beliefs"][time_step][1]

    return beliefs_at_steps


def update_belief_plot(frame, G, pos, node_size, labels, ax, beliefs_at_steps, time_step_text):
    ax.clear()

    color_palette = sns.color_palette("coolwarm", n_colors=5)
    belief_colors = {
        -2: color_palette[0],
        -1: color_palette[1],
        0: color_palette[2],
        1: color_palette[3],
        2: color_palette[4],
    }

    beliefs = beliefs_at_steps[frame]
    colors = [belief_colors[beliefs[str(node)]] for node in G.nodes()]

    nx.draw(G, pos, node_size=node_size, labels=labels, node_color=colors, with_labels=True, edge_color='gray', font_size=10, ax=ax)
    time_step_text.set_text(f"Day {frame + 1}")


def generate_belief_animation(network_file_path, agents_interaction_data_file_path, output_file, network_type, show_label=True, fps=1):
    with open(network_file_path, "r") as file:
        network_data = json.load(file)

    G = nx.Graph()
    G.add_nodes_from(network_data["nodes"])
    G.add_edges_from(network_data["edges"])

    labels = {}
    if show_label:
        for node in G.nodes():
            labels[node] = str(node)
    else:
        for node in G.nodes():
            labels[node] = ""

    if network_type == 'small_world':
        graph_name = "Small world network structure."
        pos = nx.shell_layout(G)
    elif network_type == 'scale_free':
        graph_name = "Scale free network structure"
        pos = nx.kamada_kawai_layout(G)
    else:
        graph_name = "Random network structure"
        pos = nx.kamada_kawai_layout(G)

    node_size = [G.degree[n] * 80 for n in G.nodes()]

    beliefs_at_steps = extract_beliefs_at_all_steps(agents_interaction_data_file_path)
    total_steps = len(beliefs_at_steps)
    print(total_steps)

    fig, ax = plt.subplots(figsize=(8, 8))
    time_step_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=14, verticalalignment='top')

    ani = FuncAnimation(fig, update_belief_plot, frames=total_steps,
                        fargs=(G, pos, node_size, labels, ax, beliefs_at_steps, time_step_text), interval=1000 / fps)

    ani.save(output_file, writer='pillow', fps=fps)


def visulize_opinions(network_file_path, agents_interaction_data_file_path, directory,
                      model_type, network_type, network_seed, show_label=True, step=0):

    with open(network_file_path, "r") as file:
        network_data = json.load(file)

    G = nx.Graph()
    G.add_nodes_from(network_data["nodes"])
    G.add_edges_from(network_data["edges"])

    beliefs_at_steps = extract_beliefs_at_all_steps(agents_interaction_data_file_path)

    opinions = beliefs_at_steps[step]
    file_prefix = f"llm_network_{network_type}_model_{model_type}_seed_{network_seed}_step_{step}.png"

    file_path = os.path.join(directory, file_prefix)

    labels = {}
    if show_label:
        for node in G.nodes():
            labels[node] = str(node)
    else:
        for node in G.nodes():
            labels[node] = ""

    if network_type == 'small_world':
        graph_name = "Small world network structure."
        pos = nx.shell_layout(G)
    elif network_type == 'scale_free':
        graph_name = "Scale free network structure"
        pos = nx.kamada_kawai_layout(G)
    else:
        graph_name = "Random network structure"
        pos = nx.kamada_kawai_layout(G)

    node_colors = [opinions[str(node)] for node in G.nodes()]

    node_size = [G.degree[n] * 100 for n in G.nodes()]

    # Define color map and normalization range for opinions
    # cmap = plt.get_cmap('coolwarm_r')  # You can experiment with other colormaps like 'plasma' or 'viridis'
    cmap = plt.get_cmap('coolwarm')
    vmin = -2  # Minimum opinion value
    vmax = 2  # Maximum opinion value

    print(show_label)
    # Create the figure for visualization
    plt.figure(figsize=(10, 10))
    nx.draw(
        G, pos, labels=labels, with_labels=show_label, node_color=node_colors,
        cmap=cmap, node_size=node_size, edge_color='gray', vmin=vmin, vmax=vmax
    )

    # plt.title("Network Visualization with Leader Nodes Highlighted and Opinion-based Coloring")
    plt.title(f"{graph_name}")
    plt.savefig(file_path)
    plt.savefig(file_path.replace(".png", ".pdf"))
    plt.show()
    plt.close()


def clear_cache():
    if os.path.exists("__pycache__"):
        shutil.rmtree("__pycache__")