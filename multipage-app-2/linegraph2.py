import pandas
import matplotlib
import matplotlib.pyplot as plt

# Create the DataFrame
df = pandas.DataFrame({"x":[2,4,6,8,10], "y":[200,400,600,800,1000]})

# Set the backend for matplotlib
matplotlib.use('TkAgg')

# Draw the line graph
plt.plot(df['x'], df['y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Line Graph')

# Save the graph as a png
plt.savefig('line_graph.png')