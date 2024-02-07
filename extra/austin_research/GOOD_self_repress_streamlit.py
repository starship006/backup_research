import streamlit as st
import pickle
import plotly.graph_objects as go
import os
import re

# Initialize session state variables
if 'model_name' not in st.session_state:
    st.session_state['model_name'] = 'gpt2-small'
if 'output_type' not in st.session_state:
    st.session_state['output_type'] = None
if 'receiving_type' not in st.session_state:
    st.session_state['receiving_type'] = None
if 'scaling' not in st.session_state:
    st.session_state['scaling'] = None
# %%

def plot_results(original_de, new_de, receive_heads, scaling, output_type, receiving_type, model_name):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=original_de.flatten(), 
            y=new_de.flatten(),
            text="",
            hovertext=receive_heads,
            mode="markers",
            marker=dict(size=2)
        )
    )

    fig.add_trace(
        go.Scatter(x=[original_de.min(), original_de.max()], 
                y=[original_de.min(), original_de.max()], 
                mode='lines', 
                name='y=x')
    )

    fig.update_layout(
        title=f"repression of heads | scaling == {scaling} | {model_name} | {output_type} | {receiving_type}",
        xaxis_title="Original Direct Effect",
        yaxis_title="New Direct Effect"
    )

    return fig
    
    
    
    
def load_results(model_name):
    try:
        with open(f"results_{model_name}.pickle", "rb") as f:
            results = pickle.load(f)
    except FileNotFoundError:
        st.error("No results found for this model.")
        return None
    return results    
    
# Load the model options and results
def get_model_names():
    model_files = [f for f in os.listdir('.') if f.startswith('results_') and f.endswith('.pickle')]
    model_names = [f[len('results_'): -len('.pickle')] for f in model_files]
    return model_names

model_options = get_model_names()
results = load_results(st.session_state['model_name'])

# User Interface
st.title("Model Analysis Tool")

# Model selection
col1, col2, col3, col4 = st.columns(4)
for i, model in enumerate(model_options):
    if (i % 4 == 0 and col1.button(model)) or (i % 4 == 1 and col2.button(model)) or \
       (i % 4 == 2 and col3.button(model)) or (i % 4 == 3 and col4.button(model)):
        st.session_state['model_name'] = model
        results = load_results(model)

# Dynamic options based on the model selected
if results:
    
    options = list(results.keys())
    scalings = {r[2] for r in options}
    
    if st.checkbox('Use Custom Head'):
        
        custom_options = set([i[1] for i in options if i[1].startswith('WITH_CUSTOM_HEAD')])
        
        def convert_to_tuple(input_text):
            # Modified pattern with capturing groups for numbers
            pattern = re.compile(r"[A-Za-z]+_[A-Za-z]+_[A-Za-z]+\[([0-9]+),\s+([0-9]+)\]", re.IGNORECASE)
            match = pattern.match(input_text)

            if match:
                # Extract numbers from the capturing groups and convert them to integers
                first_number = int(match.group(1))
                second_number = int(match.group(2))
                return (first_number, second_number)
            else:
                print(input_text)
                # Handle the case where the pattern does not match
                return None
            
        custom_options = [convert_to_tuple(i) for i in custom_options if (convert_to_tuple(i) is not None)]
        print(custom_options)
        all_layers = {i[0] for i in custom_options}
        all_heads = {i[1] for i in custom_options}
        # Dropdown for Layer selection
        layer = st.selectbox('Select Layer', range(min(all_layers), max(all_layers) + 1))  # Replace 'max_layer' with the actual max layer number

        # Dropdown for Head selection
        head = st.radio('Select Head', range(min(all_heads), max(all_heads) + 1))  # Replace 'max_head' with the actual max head number


        # Update receiving_type based on selection
        st.session_state['receiving_type'] = f'WITH_CUSTOM_HEAD[{layer}, {head}]'
        st.session_state['output_type'] = 'WITH_SAME_AS_RECEIVING_HEAD'
        st.session_state['scaling'] = st.radio("Scaling", list(scalings))
    else:
        output_types = {r[0] for r in options}
        receiving_types = {r[1] for r in options}
        

        st.session_state['output_type'] = st.radio("Output Type", list(output_types))
        st.session_state['receiving_type'] = st.radio("Receiving Type", list(receiving_types))
        st.session_state['scaling'] = st.radio("Scaling", list(scalings))

    if st.session_state['output_type'] and st.session_state['receiving_type'] and st.session_state['scaling']:
        orig_de, new_de, heads = results[(st.session_state['output_type'], st.session_state['receiving_type'], st.session_state['scaling'])]
        fig = plot_results(orig_de, new_de, heads, st.session_state['scaling'], st.session_state['output_type'], st.session_state['receiving_type'], st.session_state['model_name'])
        st.plotly_chart(fig)