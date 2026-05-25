# init_opinion_prompt
init_opinion_prompt = (
    """
    
    Given the topic '{topic}', your belief value is {belief}. Please provide your opinion, ensuring you include the keyword '{keyword}' to reflect your stance. 
    Also, explain the reasoning behind your opinion.

    Belief Values: Use the following scale to indicate your belief:
    • '-2' for firmly reject,
    • '-1' for somewhat disagree,
    • '0' for neutral or undecided,
    • '1' for somewhat agree,
    • '2' for strongly support.

    Rules:
    1. Use ONLY English text

    Output Structure (in code format), please follow it strictly, without any additional explanations or text:
    {{
    "opinion": "Your opinion on '{topic}' with clear focus on your stance. It should begin with: 'I {keyword} ...'",
    "belief": {belief},
    "reasoning": "The reasoning behind your opinion and belief. It should be less than 150 words."
    }}
    """
)

# recommendate_interact
recommendate_interact = (
    """
    You are a user recommendation system.

    The recommended user basic information (personal factors):
    - Age: {age}
    - Gender: {gender}
    - qualification: {qualification}
    - Personality traits: {traits}
    The recommended user beliefs on each topic are: {agent_topic_opinions} 

    Neighbor's profile is:
    - Age: {age1}
    - Gender: {gender1}
    - qualification: {qualification1}
    - Personality traits: {traits1}
    Neighbor's beliefs on each topic are: {neighbor_topic_opinions}

    Instruction:  
    Please consider both similarities and differences in profiles and topic opinions, and decide if meaningful interaction is possible.
    If there is sufficient alignment or potential for constructive exchange, interaction is encouraged.


    Output Format Requirement:
    Return a valid JSON object only. **No extra explanation, no additional sentences, just the JSON object.**
    If you return anything outside the JSON block, it will break the system. 

    Please strictly follow the output format below, without any additional explanations or text:  
    {{
        "Output": "Yes or No",
        "reason": "A brief explanation why the decision was made"
    }}  

    """
)


# long_memory_prompt
long_memory_prompt = (
    """
    Recap of Previous Long-Term Memory: {long_memory}
    Today's Short-Term Memory: {short_memory}

    Task:
    Using only the information in the previous long-term memory and today's short-term memory, create an updated long-term memory.

    Instructions:
    - Do not introduce any new information that is not present in the provided memories.
    - Start the updated memory with: "In my long-term memory, ..."
    - Accurately combine key details from both the long-term and short-term memories into a clear summary.
    - The final string must not contain newline (\n), tab (\t), or any non-printable characters.
    - Ensure the JSON value is in a single line and properly escaped.
    

    Output Structure (in JSON format), please follow it strictly, without any additional explanations or text, Please reply only in pure JSON format::
    {{
        "long_term_memory": "Your new, consolidated long-term memory statement."
    }}
    """
)

# reflecting_prompt
reflecting_prompt = (
    """
    Instruction
    Analyze the relevance of the following opinions to the current topic "{current_topic}" and generate a short-term memory summary. Categorize opinions into:
    1. Strong Positive Relevance - Explicitly and positively supports the current topic, promotes consensus, and reinforces the echo chamber effect
    2. Weak Positive Relevance - Slightly or indirectly supports the current topic, also reinforces the echo chamber effect to some extent
    3. Strong Negative Relevance - Explicitly and negatively opposes the current topic, increases diversity of opinions, and alleviates the echo chamber effect
    4. Weak Negative Relevance - Slightly or indirectly opposes the current topic, also alleviates the echo chamber effect to some extent
    5. Irrelevant - Completely unrelated to the current topic, distracts attention, and to some extent alleviates the echo chamber effect, similar to weak negative relevance

    Note: 
    - Opinions that are exactly the same as the current topic do not have any additional effect.
    -  - The influence on short-term memory is proportional to the number of discussions: topics with more discussions have a greater impact, while topics with fewer discussions have a smaller impact
    
    The opinions you have heard so far: {opinions}, topic classification: {topic_opinions}, topic counts: {topic_opinions_count}
    
    education_reform is Irrelevant.

    Task:
    Summarize the opinions provided to form your short-term memory, taking into account the relevance of each opinion to the current topic and the number of discussions per topic.

    Processing Rules::
    - Perform internal categorization first (not shown in final output)
    - Do not add or create information that is not present in the provided opinions.
    - Start the summary with: "In my short-term memory, ..."
    - Provide a brief and accurate summary of the opinions shared with you.
    - please follow the rules strictly
    - Adjust the current (current) opinion based on topic relevance

    Output Structure (in JSON format), please follow it strictly, without any additional explanations or text:
    {{
        "short_term_memory": "Your summarized short-term memory statement.[EXCLUSIVELY about {current_topic}]"
    }}
    """
)

# recommendate_topic_prompt
recommendate_topic_prompt = (
    """
    You are a professional topic recommendation system simulating a social environment. 
    Your goal is to recommend a suitable topic for a specific individual.

    Individual's basic information (personal factors):
    - Age: {age}
    - Gender: {gender}
    - Personality traits: {traits}
    - User's long-term memory: {memory}

    Current conversation context:
    - Available topics to choose from: {topics}

    Topic statistics:
    - Based on the historical choices of all users, the popularity of each topic is as follows:
      {topic_heat}
    - Based on this user's past choices, the topic fatigue is as follows (a higher value indicates the user has encountered the topic more frequently):
      {topic_fatigue}
    
    ---------------------
    Task:
    - Analyze the user's personal attributes, long-term memory, current topic, and the topic metadata.
    - Consider both topic popularity and topic fatigue in your decision (with a preference for avoiding topics with high fatigue).
    - Choose one topic from the candidate list that best suits the user's current state and promotes a meaningful and diverse conversation.

    Instruction:
    - Output strictly in the following JSON format.
    - Do not include any extra explanation or commentary.

    {{
        "topic": "<please fill one topic from the list>"
    }}


    """
)

# update_belief_prompt
update_belief_prompt = (
    """
    Your previous opinion: {opinion}
    Your previous belief value: {belief}
    Your long-term memory: {long_mem}
    Belief values: '-2' for strongly oppose, '-1' for somewhat oppose, '0' for neutral, '1' for somewhat support, '2' for strongly support.

    Task:
    - The topic you need to think about right now is {current_topic}. The opinion you received is about {topic}, and you need to compare the relevance between these two topics.
    - Reflect on your opinion and belief, considering whether to maintain your stance or adjust it based on your long-term memory.
    
    Instructions:
    - Think like a human: Decide whether to hold firm in your own opinion or adapt based on the influence of the opinions you have heard.
    - Please respond with a **pure JSON object** in the following format (and nothing else):

    Output structure (in code format), without any additional reflection or text, please follow it strictly :
    {{
    "belief": Indicate your current belief value regarding the topic.
   }}

    """
)