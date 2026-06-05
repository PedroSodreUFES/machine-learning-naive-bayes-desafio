import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_diabetes = pd.read_csv("./dataset/diabetes.csv")

sns.lineplot(
  x=range(1,11),
  y=[x * 10 for x in range(1,11)],
  color="purple",
  linestyle='--'
)
plt.savefig("./dataviz/image.png")
plt.close()