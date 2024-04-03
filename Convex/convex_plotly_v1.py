import plotly.graph_objects as go
import numpy as np

file_names = ['ray_result_0.txt', 'ray_result_1.txt', 'ray_result_2.txt', 'ray_result_3.txt', 'ray_result_4.txt', 
              'ray_tool_0.txt', 'ray_tool_1.txt', 'ray_tool_2.txt', 'ray_tool_3.txt', 'ray_tool_4.txt']

ray_result_data = []
ray_tool_data = []

for file_name in file_names:
    if 'ray_result' in file_name:
        data = np.loadtxt('Convex/Pred_/'+file_name, delimiter=',')
        ray_result_data.append(data)
    elif 'ray_tool' in file_name:
        data = np.loadtxt('Convex/True_/'+file_name, delimiter=',')
        ray_tool_data.append(data)

fig = go.Figure()

fig.add_shape(type='circle',
              x0=-0.5,  
              y0=-0.5, 
              x1=0.5,  
              y1=0.5, 
              line_color='white', 
              fillcolor='LightSkyBlue',
              layer='below',
              opacity=0.7
)  

fig.add_shape(type='rect',
              x0=0., 
              y0=-0.5, 
              x1=0.8, 
              y1=0.5,
              fillcolor='white',
              line_color='rgba(255,255,255,0)',
              layer='below'
)

for data in ray_result_data:
    x_coords = data[:, 0]
    y_coords = data[:, 1]
    fig.add_trace(go.Scatter(x=x_coords[:2], y=y_coords[:2], mode='lines', line=dict(width=6, color='black')))
    fig.add_trace(go.Scatter(x=x_coords[2:], y=y_coords[2:], mode='lines', line=dict(width=6, color='red')))
    

for data in ray_tool_data:
    x_coords = data[:, 0]
    y_coords = data[:, 1]
    fig.add_trace(go.Scatter(x=x_coords[:2], y=y_coords[:2], mode='lines', line=dict(width=6, dash='dash', color='blue')))
    fig.add_trace(go.Scatter(x=x_coords[2:], y=y_coords[2:], mode='lines', line=dict(width=6, dash='dash', color='blue')))


fig.update_xaxes(range=[-4, 4], showgrid=False, zeroline=False, showticklabels=False)
fig.update_yaxes(range=[-1, 1], showgrid=False, zeroline=False, showticklabels=False)

# 배경을 하얀색으로 변경
fig.update_layout(plot_bgcolor='white')
fig.show()