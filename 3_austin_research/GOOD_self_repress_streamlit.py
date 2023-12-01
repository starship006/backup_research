# %%
import streamlit as st
import pickle
import plotly.graph_objects as go
# %%

def plot_results(original_de, new_de, receive_heads, scaling, output_type, receiving_type):
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
        title=f"repression of heads | scaling == {scaling} | {output_type} | {receiving_type}",
        xaxis_title="Original Direct Effect",
        yaxis_title="New Direct Effect"
    )

    return fig
    
# %%
    
def load_results(model_name):
    with open(f'results_{model_name}.pickle', 'rb') as f:
        return pickle.load(f)

model_name = st.sidebar.selectbox("Select model", ["pythia-160m", "gpt2-small", "gpt-neo-125m", "gpt2-large",
                                                   "gpt2-medium", "pythia-410m", "stanford-gpt2-medium-a", "stanford-gpt2-small-a"])
results = load_results(model_name) 

output_type = st.sidebar.selectbox("Output type", options={r[0] for r in results.keys()})
receiving_type = st.sidebar.selectbox("Receiving type", options={r[1] for r in results.keys()}) 
scaling = st.sidebar.selectbox("Scaling", options={r[2] for r in results.keys()})

orig_de, new_de, heads = results[(output_type, receiving_type, scaling)]

st.subheader(f"Results for {output_type}, {receiving_type}, scaling={scaling}")
fig = plot_results(orig_de, new_de, heads, scaling, output_type, receiving_type)

st.plotly_chart(fig)


# %%
