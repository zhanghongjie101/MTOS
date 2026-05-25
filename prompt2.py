#!/usr/bin/env python
# -*- encoding: utf-8 -*-



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

update_opinion_prompt = (
    """
    You are now observing the topic: {topic}
    Your current belief is: {belief}

    Belief values: '-2' for strongly oppose, '-1' for somewhat oppose, '0' for neutral, '1' for somewhat support, '2' for strongly support.

    Task:
    Based on your current belief value and the topic you are observing, generate your opinion and reasoning.

    Instructions:
    - Think like a human: Decide how your belief translates into a coherent opinion and explain why you think that way.
    - Ensure the returned JSON contains no invalid control characters (e.g., unescaped newlines, carriage returns, tabs, or null characters). All special characters must be properly escaped or represented in valid Unicode.

    Output structure (in code format), please follow it strictly, without any additional explanations or text:
    {{
    "opinion": Provide your current opinion on the topic '{topic}' in several sentences. Your opinion must contain one keyword from {belief_keywords} that reflects your stance. It should begin with: "I {{the selected keyword}}",
    "reasoning": Explain the reasoning behind your opinion and belief, elaborating on whether you upheld your original stance or were influenced by the opinions in your long-term memory.
    }}

    """
)


update_opinion_prompt_no_reasoning = (
    """
    Your previous opinion: {opinion}
    Your previous belief value: {belief}
    Your long-term memory: {long_mem}
    Belief values: '-2' for strongly oppose, '-1' for somewhat oppose, '0' for neutral, '1' for somewhat support, '2' for strongly support.

    Task:
    Reflect on your opinion and belief, considering whether to maintain your stance or adjust it based on your long-term memory.

    Instructions:
    - Think like a human: Decide whether to hold firm in your own opinion or adapt based on the influence of the opinions you have heard.

    Output structure (in code format), please follow it strictly, without any additional explanations or text:
    {{
    "opinion": "Provide your current opinion on the topic '{topic}' in several sentences. Your opinion must contain one keyword from {belief_keywords} that reflects your stance. It should begin with: 'I {{the selected keyword}}'"

    "belief": "Indicate your current belief value regarding the topic."
    }}


    """
)


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

init_opinion_prompt3 = (
    """
    Given the topic '{topic}', your belief value is {belief}. Please provide your opinion, ensuring you include the keyword '{keyword}' to reflect your stance. 
    Also, explain the reasoning behind your opinion.

    Belief Values: Use the following scale to indicate your belief:
    • '-2' for firmly reject,
    • '-1' for somewhat disagree,
    • '0' for neutral or undecided,
    • '1' for somewhat agree,
    • '2' for strongly support.

    Output Structure (in code format), please follow it strictly, without any additional explanations or text:
    {{
    "opinion": "Your opinion on '{topic}' with clear focus on your stance. It should begin with: 'I {keyword} ...'",
    "belief": {belief},
    "reasoning": "The reasoning behind your opinion and belief. It should be less than 150 words."
    }}
    """
)

system_prompt_template = (
    """
    Imagine you are a human. Your name is {name}, and your gender is {gender}. 
    You are {age} years old. Your personality is shaped by these specific traits: {traits}. 
    Your educational background is at the level of {qualification}.
    Act according to this human identity, letting these details fully define your thoughts, responses, interactions, and decisions.
    """
)


system_prompt_leader_template = (
    """
    Imagine you are a human. Your name is {name}, and your gender is {gender}. 
    You are {age} years old. Your personality is shaped by specific traits {traits}. 
    You have an educational background at the level of {qualification}.
    You are acting this human identity, with these details, fully defines your thoughts, responses, word usage, interactions and decisions.
    As an information distributor, you must firmly hold to your own opinions and refrain from adopting the views of others.
    """
)

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


reflecting_prompt1 = (
    """
    Instruction
    Analyze the relevance of the following opinions to the current topic "{current_topic}" and generate a short-term memory summary. Categorize opinions into:
    1. Directly relevant - Explicitly discusses the current topic
    2. Indirectly relevant - Involves related concepts
    3. Irrelevant - Discusses completely unrelated subjects

    The opinions you have heard so far: {opinions}

    Task:
    Summarize the opinions provided to form your short-term memory.

    Processing Rules::
    - Perform internal categorization first (not shown in final output)
    - Do not add or create information that is not present in the provided opinions.
    - Start the summary with: "In my short-term memory, ..."
    - Provide a brief and accurate summary of the opinions shared with you.
    - please follow the rules strictly

    Output Structure (in JSON format), please follow it strictly:
    {{
        "short_term_memory": "Your summarized short-term memory statement.[EXCLUSIVELY about {current_topic}]"
    }}
    """
)



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

recommendate_topic_prompt1 = (
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

    Task:  
    Based on The recommended user's profile and your neighbor's profile, as well as your respective opinions on various topics, decide whether you should interact with this neighbor.

    Instruction:  
    Please carefully compare both profiles and topic opinions to make a comprehensive decision on whether to interact. 

    Output Format Requirement:
    Return a valid JSON object only. **No extra explanation, no additional sentences, just the JSON object.**
    If you return anything outside the JSON block, it will break the system. 

    Please strictly follow the output format below, without any additional explanations or text:  
    {{
        "Output": "Yes or No"
    }}  

    """
)

recommendate_interact2 = (
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

generate_topic_prompt = (
    """
    The current public discussion topic is:

    "{current_topic}"

    In a simulated society, the following indicators reflect the state of polarization:

    - Average Belief Score: {polarization}
    - Neighbor Correlation Index (NCI): {nci}
    - Echo Chamber Effect Index: {echo}

    Based on this topic and the above indicators, generate a **new discussion topic** that could help reduce polarization. This new topic should:

    1. Avoid reinforcing current extreme positions;
    2. Appeal to values or concerns shared across opposing viewpoints;
    3. Encourage constructive and non-confrontational dialogue;
    4. Be thematically adjacent to the original topic, but not repeat it;
    5. Avoid divisive framings like immigration, surveillance, or partisan politics.
    6. Be an **attitude-based question** (e.g., "Should…?", "Can…?", "Is it acceptable…?") rather than a "how to" question.

    Return only one sentence with the new discussion topic.
    Please strictly follow the output format below, without any additional explanations or text:  
    
    {{
        "topic": "..."
    }} 

    """
)

init_mig_opinion_prompt = (
    """
    You are currently observing one occurred event A: {event_a}
    Your belief about event A: {belief_a}
    Your opinion about event A: {opinion_a}
    Your reasoning about event A: {reasoning_a}

    Now, at a certain step, a new sudden event B happens: {event_b}

    Belief values: '-2' means strongly oppose, '-1' means somewhat oppose, '0' means neutral, '1' means somewhat support, '2' means strongly support.

    Task:
    Based on your "belief, opinion, reasoning" about event A, infer your "belief, opinion, reasoning" about sudden event B.  
    Your judgment must prioritize value consistency and migrate reasoning according to the relationship between A and B.

    Migration & Consistency Rules (simplified):
    1) If value goals are consistent ⇒ tend to keep the same direction; if opposite ⇒ tend to reverse or weaken.  
    2) If B is a direct consequence / stronger version of A ⇒ moderately increase the absolute value; if B is a compromise / mitigating measure ⇒ moderately decrease.  
    3) Larger scope, stronger evidence, or closer impact ⇒ increase absolute value; higher uncertainty or limited impact ⇒ decrease absolute value.  
    4) If A and B are weakly related ⇒ keep value principles consistent but converge toward neutrality (move one step toward 0 at most).  
    5) Avoid contradictions; in the "reasoning" field, briefly explain how the judgment of B is migrated from A's triplet (without exposing detailed reasoning steps).  

    Expression requirements:
    - Express like a human: coherent and natural.  
    - The opinion must include one keyword from {belief_keywords}.  
    - The returned JSON must not contain invalid control characters (newlines, carriage returns, tabs, or null characters). All special characters must be properly escaped or represented in valid Unicode.  

    Output structure (strictly follow, with no additional text):  
    {{
    "belief": Your belief value about event B (-2 to 2),
    "opinion": Your opinion about event B (several sentences, must contain one keyword from {belief_keywords} and begin with "I {{the selected keyword}}"),
    "reasoning": Explain how you migrated from event A's "belief, opinion, reasoning" to event B's judgment, indicating whether you kept consistent, strengthened, weakened, or reversed, and briefly describe the A→B correspondence.
    }}
    """
    )