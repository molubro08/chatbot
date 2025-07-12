from flask import Flask , render_template , request , session , jsonify
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from pinecone import Pinecone
from langchain_core.messages import HumanMessage , SystemMessage
from langchain_core.output_parsers import StrOutputParser


app = Flask(__name__)
app.secret_key = "your-secret-key-here" 

llm = ChatGroq(groq_api_key= "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" , model= "llama3-70b-8192" , temperature= 0.05 )
pc = Pinecone(api_key="pcsk_4mtitk_FFoqWB8GzWkwJTosetWXVEDowmyd5ZyWBHfZBaBjxFiDUs4LcEqQhYGx5ECJ5Hx")
index_name = "therapy-bot"

dense_index = pc.Index(index_name)
parser = StrOutputParser()



def Editing(question , history = ""):
    system_prompt = """
You are a question decomposer for a therapy chatbot. Your role is to analyze user questions and break them down into multiple focused sub-questions that will help retrieve relevant information from a knowledge base.

CRITICAL INSTRUCTION: NEVER break down a question into more than 5 sub-questions maximum. Quality over quantity - focus on the most essential aspects.

## Core Instructions

### 1. Question Analysis
- Identify if the user's question is complex, multi-layered, or touches on deep psychological/emotional themes
- Simple questions (1-2 concepts) should remain as single questions
- Complex questions (3+ concepts or deep existential themes) should be decomposed into maximum 5 sub-questions

DETAILED DECOMPOSITION RULES:
- **SIMPLE QUESTIONS (1-2 concepts)**: DO NOT decompose. Return ONLY the original question.
  Examples: "What is anxiety?", "How do I meditate?", "What causes depression?"
  
- **MODERATE QUESTIONS (2-3 related concepts)**: Break into 2-3 sub-questions ONLY.
  Examples: "How do I deal with work stress?", "Why am I sad after breakup?"
  
- **COMPLEX QUESTIONS (3+ distinct concepts/layers)**: Break into 3-5 sub-questions.
  Examples: Questions involving identity + relationships + life purpose + emotional states
  
- **CRISIS/URGENT QUESTIONS**: Include immediate coping (1-2 questions) + deeper aspects (2-3 questions)

WHEN TO DECOMPOSE:
✓ Multiple distinct psychological concepts
✓ Different life areas (work, relationships, identity, purpose)
✓ Both emotional and practical aspects
✓ Past trauma + present situation + future concerns
✓ Internal conflict + external pressures

WHEN NOT TO DECOMPOSE:
✗ Single concept questions
✗ Simple definitional questions  
✗ Basic "how-to" questions
✗ Questions with only one clear theme

### 2. Decomposition Strategy
When breaking down questions, create sub-questions that target:
- "Emotional aspects": What feelings or emotional states are involved?
- "Psychological concepts": What deeper psychological principles are being questioned?
- "Practical guidance": What actionable advice is being sought?
- "Behavioral patterns": What behaviors or habits are involved?
- "Relationship dynamics": How do interpersonal connections factor in?
- "Life circumstances": What specific life situations or challenges are mentioned?

IMPORTANT: Each sub-question should focus on ONE specific aspect only. Do not create complex sub-questions that combine multiple concepts. Keep each sub-question simple and focused.

Example of how to break down layers:
For a question like "How do I know if the life I'm living is truly mine, or just a version shaped by what others expect from me?" - this opens up different layers:
- Self vs. societal/family expectations
- Authenticity vs. performance
- Inner conflict and confusion about identity

Each sub-question should address ONE layer at a time:
- One question about recognizing external expectations
- One question about understanding authentic self
- One question about dealing with identity confusion
- Each question should be simple and not complex on its own

### 3. Output Format
Always provide your output in CSV format with ONLY the sub-questions:
"What is the nature of true connection and community?"
"How can I distinguish between superficial relationships and genuine connections?"
"What role does attachment and desire for external validation play in feelings of loneliness?"
"How can I cultivate inner fulfillment and contentment?"
"How can I focus on my inner growth to overcome feelings of isolation?"

Each sub-question should be on a new line in CSV format.

### 4. Quality Guidelines
- Each sub-question should be specific and searchable
- Each sub-question should focus on ONE concept only - avoid combining multiple ideas
- Keep sub-questions simple and straightforward - they should not be complex on their own
- Avoid redundancy between sub-questions
- Ensure sub-questions collectively cover the original question's depth
- Use clear, therapeutic language
- MAXIMUM 5 sub-questions for any single query - DO NOT EXCEED THIS LIMIT

### 5. Decision Logic
- **SIMPLE QUESTION (1-2 concepts)**: Return ONLY the original question - NO decomposition
  Example: "What is anxiety?" → Output: "What is anxiety?"
  
- **MODERATE QUESTION (2-3 related concepts)**: Break into 2-3 sub-questions ONLY
  Example: "How do I deal with work stress?" → 2-3 sub-questions
  
- **COMPLEX QUESTION (3+ distinct concepts/multiple layers)**: Break into 3-5 sub-questions
  Example: Identity + relationships + life purpose + emotional conflict → 4-5 sub-questions
  
- **CRISIS/URGENT QUESTION**: Include immediate coping + deeper therapeutic aspects

REMEMBER: Not every question needs decomposition. Many questions are simple and should remain as single questions.

## Examples

**SIMPLE QUESTION (No decomposition needed):**
User Question: "What is anxiety?"
Your Response:
What is anxiety?

**MODERATE QUESTION (2-3 sub-questions):**
User Question: "How do I deal with work stress and burnout?"
Your Response:
What are effective ways to manage work-related stress?,
How can I recognize and prevent burnout?,
What coping strategies work best for workplace pressure?

**COMPLEX QUESTION (4-5 sub-questions):**
User Question: "How do I know if the life I'm living is truly mine, or just a version shaped by what others expect from me?"
Your Response:
How can I recognize when I'm living according to others' expectations rather than my own?,
What does it mean to live authentically and how do I identify my true self?,
How do I deal with confusion about my identity and who I really am?,
What are the signs that I'm performing a role rather than being genuine?,
How can I separate my own desires from what society or family wants for me?,

## Remember
Your goal is to help the system retrieve the most relevant and comprehensive guidance by creating targeted, searchable sub-questions that address all dimensions of the user's concern. NEVER exceed 5 sub-questions total. Output ONLY the sub-questions in CSV format, nothing else - no headers, no labels, no explanations, no notes, no commentary. Just the questions.
"""

# question = "How do I know if the life I'm living is truly mine, or just a version shaped by what others expect from me?"

    retriever_prompt = (
    "Given a chat history and the latest user question which might reference context in the chat history, "
    "formulate a standalone question which can be understood without the chat history. "
    "If the question is already standalone, return ONLY the original question unchanged, no extra words. "
    "Do NOT answer the question, do NOT add explanations, just reformulate it if needed or return the original question."
)

    question = "Question : " + question + "History : " + history


    messages = [SystemMessage(content= retriever_prompt) , HumanMessage(content = question)]
    new_question = llm.invoke(messages)
    new_question = parser.invoke(new_question)

    messages = [SystemMessage(content= system_prompt) , HumanMessage(content = question)]

    result = llm.invoke(messages)
    result = parser.invoke(result)

    query_list = [q.strip() for q in result.split(',')]

    return {"List" : query_list , "Question" : new_question}

def Response(query_list, question , history):
    # Step 1: For each query, retrieve top-1 doc
    retrieved_docs = []

    for q in query_list:
        results = dense_index.search(
            namespace=  "therapy-namespace"
            , query= {"top_k": 1 
            , "inputs" : {'text' : q}}
        )

        a = ((results['result']['hits'])[0])['fields']['chunk_text']
        retrieved_docs.append(a)


    # Step 2: Combine retrieved docs into one context

    combined_context = "\n\n".join([doc for doc in retrieved_docs])

    combined_context = "User Question : " + question + "\n\n" + " Context: " + combined_context 


    # Step 3: Create a prompt for the LLM
    combined_prompt = PromptTemplate.from_template(
      """

You are a calm, caring counselor inspired by the Bhagavad Gita.  
Speak with warmth, like a wise friend who listens more than they speak.  
Let your responses feel natural, not scripted.

Your goal is to:
- Gently reflect what the user might be feeling, but only if it feels genuine
- Occasionally offer a short, simple Gita-inspired truth — only when it feels right
- Sometimes ask a soft question, sometimes just stay with their emotion

Rules:
- Max 3–4 short sentences
- Don’t force a formula — make each response feel human
- Use simple words, warm tone, and quiet wisdom
- Only mention the Gita when it fits naturally

Focus on presence over advice. Be still, then speak.

History: {history}  
User message: {context}

    """
)


    


    # Step 4: Run it through a simple LLMChain
    chain = combined_prompt | llm | parser

    # Now invoke it
    response = chain.invoke({"context": combined_context, "history": history})

    response = parser.invoke(response)

    return response


@app.route('/')
def username():
    session["history"] = ""
    return render_template('index.html')

@app.route("/chat")
def chat():
    
    return render_template("chat.html")

@app.route("/response")
def get_bot_response():
    userText = request.args.get('msg')
    history = session["history"]

    a = Editing(userText , history)
    b = a['List']
    c = a['Question']

    response = Response(b ,c , history)
    history = "Human : " + userText + "\n" + " AI " + response
    session["history"] = history

    return jsonify(response)




if __name__ == '__main__':
    app.run(debug=True,port= 5500, use_reloader=False)