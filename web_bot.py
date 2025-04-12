# llm_chatbot_app.py

import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.memory import ConversationBufferMemory
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import AIMessage, HumanMessage
import re
import warnings
import os # Good practice to import os if using env vars, though not strictly needed here

# --- Configuration ---
# Allow overriding Ollama URL via environment variable for flexibility
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = "llama3.2" # Consider making this configurable too if needed
AGENT_PROMPT_REPO = "hwchase17/react-chat"

# --- Suppress Specific Langchain Warning ---
# Suppresses a specific warning related to ConversationBufferMemory output keys,
# often encountered with older langchain versions. Upgrading langchain is the ideal fix.
warnings.filterwarnings(
    "ignore",
    message=".*'ConversationBufferMemory' got multiple output keys.*",
    category=UserWarning,
    module="langchain.memory.chat_memory",
)

# --- Streamlit Page Setup ---
st.set_page_config(page_title=f"Chat with {OLLAMA_MODEL}", layout="wide")
st.title(f"🤖 Chatbot powered by Ollama ({OLLAMA_MODEL}) & Langchain")
st.caption("This chatbot can use DuckDuckGo to search the web for recent information.")

# --- Sidebar Settings ---
st.sidebar.title("⚙️ Settings")

# Slider for number of search results
# Use session state to preserve the slider's value across reruns.
# Initialize with a default value if not already set.
if 'max_search_results' not in st.session_state:
    st.session_state.max_search_results = 3 # Default to 3 search results

# Create the slider widget. Its value is automatically stored in
# st.session_state.max_search_results due to the 'key' argument.
st.sidebar.slider(
    "Number of Search Results",
    min_value=1,
    max_value=10, # You can adjust this maximum limit
    key="max_search_results", # Links the slider directly to the session state variable
    step=1,
    help="Select the maximum number of search results the agent should retrieve and display." # Updated help text
)
# Get the current value from session state (which is updated by the slider)
current_max_results = st.session_state.max_search_results

# Clear History Button
if st.sidebar.button("Clear Chat History"):
    # Clear displayed messages
    st.session_state.messages = []
    # Re-initialize the conversation memory
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
    # Crucially, remove the agent executor from session state.
    # This forces it to be recreated with the fresh memory and current tool settings
    # on the next run.
    if "agent_executor" in st.session_state:
        del st.session_state.agent_executor
    # Optionally clear other related states if needed
    # if "search_tool" in st.session_state: del st.session_state.search_tool
    # if "llm_initialized" in st.session_state: del st.session_state.llm_initialized
    st.rerun() # Rerun the app to reflect the cleared state

# --- Core Application Logic ---

# 1. Initialize LLM (only once or if connection fails)
# Check if LLM is already initialized and stored in session state
if "llm_initialized" not in st.session_state or not st.session_state.llm_initialized:
    with st.spinner(f"Connecting to Ollama model '{OLLAMA_MODEL}'..."):
        try:
            # Attempt to create and test the LLM connection
            llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.7)
            llm.invoke("Hello!") # Simple test call to ensure connectivity and model availability
            # Store the successful LLM instance and flag in session state
            st.session_state.llm = llm
            st.session_state.llm_initialized = True
        except Exception as e:
            # Display error and stop if connection fails on the first attempt
            st.error(f"Error connecting to Ollama ('{OLLAMA_MODEL}' at {OLLAMA_BASE_URL}): {e}")
            st.error("Please ensure the Ollama server is running and the model is available.")
            # Display minimal sidebar elements even if LLM fails
            st.sidebar.markdown("---")
            st.sidebar.warning("LLM connection failed. Cannot initialize chatbot.")
            st.stop() # Stop script execution
else:
    # Retrieve the previously initialized LLM from session state
    llm = st.session_state.llm

# 2. Initialize Tools (conditionally based on slider value)
# This section ensures the search tool is updated if the slider value changes.
tool_needs_update = False
# Check if the tool doesn't exist yet
if "search_tool" not in st.session_state:
    tool_needs_update = True
# Check if the tool exists but its max_results setting doesn't match the current slider value
elif st.session_state.search_tool.max_results != current_max_results:
    tool_needs_update = True

# If the tool needs to be created or updated:
if tool_needs_update:
    # Create a new instance of the search tool with the current max_results value
    st.session_state.search_tool = DuckDuckGoSearchResults(
        name="duckduckgo_search_results",
        max_results=current_max_results # Use the value from the slider via session state
    )
    # IMPORTANT: If the tool configuration changes, the agent executor *must* be recreated
    # because it holds a reference to the specific tool list used during its creation.
    # Delete the existing agent executor from session state to force its recreation later.
    if "agent_executor" in st.session_state:
        print(f"DEBUG: Recreating agent executor because max_results changed to {current_max_results}") # Optional debug print
        del st.session_state.agent_executor

# Ensure the 'tools' list always uses the potentially updated tool from session state
tools = [st.session_state.search_tool]

# 3. Initialize Memory (only once, unless cleared)
# Checks if memory exists in session state, initializes if not.
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )

# 4. Initialize List for UI Messages (only once)
# This list stores messages purely for display in the Streamlit UI.
if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. Initialize Agent Components (prompt, agent)
# These are relatively lightweight and can be recreated on each run.
# Pulling the prompt template from the hub.
prompt_template = hub.pull(AGENT_PROMPT_REPO)
# Creating the agent function using the LLM, the current tools list, and the prompt.
agent = create_react_agent(llm, tools, prompt_template)

# 6. Initialize Agent Executor (only if missing or forced by tool/memory update)
# Checks if the executor needs to be created (either first run or after tool/memory change).
if "agent_executor" not in st.session_state:
    # Creates the agent executor, which manages the agent's interaction loop.
    st.session_state.agent_executor = AgentExecutor(
        agent=agent,
        tools=tools, # Pass the potentially updated tools list
        memory=st.session_state.memory, # Pass the conversation memory
        verbose=True, # Set to True to see agent's internal steps in the console/log
        handle_parsing_errors=True, # Attempt to gracefully handle LLM output parsing issues
        return_intermediate_steps=True, # Required to access tool outputs (like search results)
    )

# Retrieve the current agent executor from session state for use in this run
agent_executor = st.session_state.agent_executor


# --- Chat Interface Logic ---

# Display past chat messages from the session state list
for message_info in st.session_state.messages:
    with st.chat_message(message_info["role"]):
        # Display the message content
        st.markdown(message_info["content"])
        # If it's an assistant message and has sources, display them
        if message_info["role"] == "assistant" and "sources" in message_info and message_info["sources"]:
            st.caption("Sources:")
            processed_sources = []
            # Iterate through stored sources (expected to be list of dicts or strings)
            for source in message_info["sources"]:
                # Handle dictionary sources (preferred format)
                if isinstance(source, dict) and "link" in source:
                    title = source.get('title', '') # Get title or empty string if missing
                    link = source['link']
                    # Use title if available and meaningful, otherwise use link
                    display_text = title if title and title != "Source Link" else link
                    processed_sources.append(f"- [{display_text}]({link})")
                # Handle simple string URL sources (fallback)
                elif isinstance(source, str) and source.startswith("http"):
                     # Display the URL string as the clickable link text
                     processed_sources.append(f"- [{source}]({source})")
                # Silently skip if source format is unexpected
            # Display the formatted source links if any were processed
            if processed_sources:
                 st.markdown("\n".join(processed_sources), unsafe_allow_html=True)


# Get user input from the chat input widget at the bottom
if user_query := st.chat_input("What can I help you with?"):
    # Add user's message to the display list and session state
    st.session_state.messages.append({"role": "user", "content": user_query})
    # Display user's message immediately
    with st.chat_message("user"):
        st.markdown(user_query)

    # Prepare the input for the agent executor
    agent_input = {"input": user_query} # The ReAct prompt expects 'input'

    # Display a thinking spinner while the agent processes the request
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # *** Execute the agent ***
                response = agent_executor.invoke(agent_input)

                # Extract the final answer from the agent's response
                agent_response_content = response.get("output", "Sorry, I couldn't generate a response.")

                # --- Extract source information from intermediate steps ---
                source_links = set() # Use a set to automatically handle duplicate links
                source_details = [] # Store details (link, title) for display
                # *** Get the actual limit set by the user from the slider ***
                results_limit = st.session_state.max_search_results

                # Check if the agent provided intermediate steps (tool usage)
                if "intermediate_steps" in response:
                    # Iterate through the (action, observation) pairs
                    for step in response["intermediate_steps"]:
                        # *** Check if we have already reached the desired number of unique sources ***
                        if len(source_links) >= results_limit:
                            break # Stop processing further steps if limit is reached

                        action, observation = step
                        # Check if the action taken was using the configured search tool
                        if hasattr(action, "tool") and action.tool == st.session_state.search_tool.name:
                            # Process the observation returned by the search tool
                            if isinstance(observation, list): # Expected format is list of dicts
                                for result in observation:
                                    # *** Check limit again before processing each result from the tool ***
                                    if len(source_links) >= results_limit:
                                        break # Stop processing results in this observation if limit reached

                                    if isinstance(result, dict) and "link" in result:
                                        link = result["link"]
                                        # Add to set to check for duplicates; add details ONLY if new AND limit not reached
                                        if link not in source_links:
                                            source_links.add(link)
                                            source_details.append({
                                                "link": link,
                                                "title": result.get("title", "Source Link")
                                            })
                                # Exit the outer loop if the limit was reached within this inner loop
                                if len(source_links) >= results_limit:
                                    break
                            elif isinstance(observation, str): # Fallback if observation is just a string
                                found_urls = re.findall(r"http[s]?://\S+", observation)
                                for url in found_urls:
                                     # *** Check limit again before processing each found URL ***
                                     if len(source_links) >= results_limit:
                                        break # Stop processing URLs if limit reached

                                     if url not in source_links:
                                        source_links.add(url)
                                        source_details.append({"link": url, "title": "Source Link"})
                                # Exit the outer loop if the limit was reached within this inner loop
                                if len(source_links) >= results_limit:
                                    break

                # --- Display the final response and sources ---
                # Display the agent's final text response
                st.markdown(agent_response_content)

                # Display the collected source links, if any (now strictly respects the limit)
                if source_details:
                    st.caption("Sources:")
                    source_markdown = []
                    # Sort sources for consistent display order
                    for detail in sorted(source_details, key=lambda x: x["link"]):
                        title = detail.get('title', '') # Get title or empty string
                        link = detail['link']
                        # Use title if available and meaningful, otherwise use link
                        display_text = title if title and title != "Source Link" else link
                        source_markdown.append(f"- [{display_text}]({link})")
                    st.markdown("\n".join(source_markdown), unsafe_allow_html=True)

                # Add the assistant's response and sources to the message history for display
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": agent_response_content,
                    # Store the structured source details (which now respects the limit)
                    "sources": source_details
                })

            except Exception as e:
                # Handle errors during agent execution
                error_message = f"An error occurred during agent execution: {e}"
                st.error(error_message)
                # Add an error message to the chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Error: Could not complete the request. Details: {e}",
                    "sources": [] # No sources in case of error
                })
